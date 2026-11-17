import json
from pathlib import Path
from PIL import Image
VARIABLE_DEFINITIONS = {'MAP': 'mean annual precipitation', 'MAT': 'mean annual temperature'}
METHOD_DEFINITIONS = {
    'Modern Analogue': 'modern analogue technique',
    'RF': 'random forest',
    'ExtraTrees': 'extremely randomized trees',
    'Boosting': 'histogram gradient boosting',
}
LEGACY_VARIABLES = {
    'Annual precipitation (mm/year)': 'MAP',
    'Annual temperature (deg C)': 'MAT',
}
def normalize_cache(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key=='variable_definitions':
                result[key]=dict(VARIABLE_DEFINITIONS)
                continue
            if key=='method_definitions':
                result[key]=dict(METHOD_DEFINITIONS)
                continue
            if key=='climate_variable':
                result[key]=LEGACY_VARIABLES.get(item,item)
                continue
            if key in LEGACY_VARIABLES:
                new_key=LEGACY_VARIABLES[key]
            elif key=='MAT' and isinstance(item,dict) and 'cores' in item:
                new_key='MAT'
            else:
                new_key='Modern Analogue' if key=='MAT' else key
            result[new_key]=normalize_cache(item)
        return result
    if isinstance(value,list):
        return [normalize_cache(item) for item in value]
    if isinstance(value,str):
        if value=='MAT':return 'Modern Analogue'
        return value.replace('Rainfall MAT','Rainfall Modern Analogue').replace('Random forest','RF').replace('Extra trees','ExtraTrees').replace('Gradient boosting','Boosting')
    return value
def load_cached(path):
    with Image.open(Path(path)) as image:
        result=normalize_cache(json.loads(image.info['Description']))
    return result
def add_definitions(result,variable=None):
    result['variable_definitions']=dict(VARIABLE_DEFINITIONS)
    result['method_definitions']=dict(METHOD_DEFINITIONS)
    if variable is not None:result['climate_variable']=variable
    return result
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import json, sys, importlib.util, itertools
import numpy as np
import numpy as np
from scipy.ndimage import gaussian_filter1d
BAND_LABEL = '95% bin-resampling band'
BAND_NOTE = ('Pointwise 2.5th and 97.5th percentiles from 2000 independent exponential-weight '
             'bootstrap draws of occupied 0.25 Ma bins within each core. Ages, fitted models, '
             'core membership and temporal support are fixed. Each draw repeats within-core '
             'smoothing, equal-core averaging, gap interpolation and regional smoothing.')
def trend_band(series, grid, curve_age, center, core_sigma, region_sigma):
    rng = np.random.default_rng(20261115)
    draws = 2000
    total = np.zeros((len(grid), draws + 1))
    support = np.zeros(len(grid), dtype=int)
    for age, value in series:
        valid = np.isfinite(age) & np.isfinite(value)
        if not valid.any():
            continue
        age, value = age[valid], value[valid]
        _, index = np.unique(np.floor(age / .25), return_inverse=True)
        counts = np.bincount(index)
        bin_age = np.bincount(index, weights=age) / counts
        bin_mean = np.bincount(index, weights=value) / counts
        distance = grid[:, None] - bin_age
        kernel = np.exp(-.5 * (distance / core_sigma)**2)
        kernel[np.abs(distance) > round(3 * core_sigma, 8)] = 0
        active = (grid >= age.min()) & (grid <= age.max()) & (kernel.sum(axis=1) > 0)
        weights = np.ones((len(counts), draws + 1))
        weights[:, 1:] = rng.exponential(size=(len(counts), draws))
        k = kernel[active]
        total[active] += ((k * bin_mean) @ weights) / (k @ weights)
        support[active] += 1
    valid = support > 0
    if not valid.any():
        assert not len(curve_age)
        return np.empty(0), np.empty(0)
    x = grid[valid]
    values = total[valid] / support[valid, None]
    assert curve_age[0] == x[0] and curve_age[-1] == x[-1]
    filled = np.column_stack([np.interp(curve_age, x, v) for v in values.T])
    smoothed = gaussian_filter1d(filled, region_sigma / (grid[1] - grid[0]),
                                 axis=0, mode='nearest', truncate=3)
    np.testing.assert_allclose(smoothed[:, 0], center, rtol=1e-8, atol=1e-8)
    return np.quantile(smoothed[:, 1:], [.025, .975], axis=1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.spatial.distance import cdist
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if '--root' in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index('--root')+1]).resolve()
REGIONS = [['KC-1', 'SG-1b', 'SG-1'], ['DHG', 'YH2', 'HTTL2']]
NAMES = ['Modern Analogue', 'WA', 'WA-PLS', 'RF', 'ExtraTrees', 'Boosting']
ENSEMBLE_SIZE = 5
COMB = list(itertools.combinations(range(len(NAMES)), ENSEMBLE_SIZE))
def module(i):
    p = ROOT / f'Supplementary Fig {i-1}.py'
    spec = importlib.util.spec_from_file_location('source' + str(i), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
def traditional(m, a, y, q, rain=False):
    neighbors = m.choices['Modern Analogue'][0][1] if rain else 5
    components = m.choices['WA-PLS'][0][2] if rain else 2
    out = []
    m.METHOD = 'Modern Analogue'
    m.ACTIVE = np.array([m.CANDS.index(('Modern Analogue', neighbors))])
    out.append(m.predict(a, y, q, np.ones(len(a), bool))[:, 0])
    out.append(m.standalone_wa(a, y, q))
    out.append(m.wa(m.norm(np.sqrt(a)), y, m.norm(np.sqrt(q)), components)[:, -1])
    return out
def fit(m, a, y, q):
    out = traditional(m, a, y, q)
    for cls in [RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor]:
        model = cls(n_estimators=200, max_features=0.5, min_samples_leaf=5, n_jobs=4, random_state=20261027) if cls != HistGradientBoostingRegressor else cls(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=10, learning_rate=0.05, early_stopping=False, random_state=20261027)
        out.append(model.fit(a, y).predict(q))
    return np.column_stack(out)
def choose(records):
    scores = []
    bad = []
    for c, (y, P) in records.items():
        ok = np.isfinite(P).all(1)
        y = y[ok]
        P = P[ok]
        e = P - y[:, None]
        H = e.T @ e / len(y) / np.var(y)
        v = np.array([H[np.ix_(u, u)].mean() for u in COMB])
        scores.append(np.sqrt(v))
        bad.append(v >= 1)
    loss = np.mean(scores, 0)
    fail = np.sum(bad, 0)
    pool = np.flatnonzero(fail == fail.min())
    i = pool[np.argmin(loss[pool])]
    return (COMB[i], float(loss[i]))
def choose_rain(records):
    losses = []
    dry_bias = []
    for y, p, dry in records.values():
        valid = np.isfinite(p).all(1)
        err = np.column_stack([p[valid][:, u].mean(1) for u in COMB]) - y[valid, None]
        losses.append(np.sqrt(np.mean(err ** 2, axis=0)) / np.std(y[valid]))
        dry_bias.append(err[dry[valid]].mean(0))
    losses = np.array(losses)
    mean = losses.mean(0)
    fail = (losses >= 1).sum(0)
    pool = np.flatnonzero(fail == fail.min())
    near = pool[mean[pool] <= 1.01 * mean[pool].min()]
    worst = np.maximum(0, np.max(dry_bias, axis=0))
    j = min(near, key=lambda i: (worst[i], mean[i], i))
    return (COMB[j], float(mean[j]))
def cell_metric(y, p, cells):
    good = np.isfinite(p) & np.isfinite(y)
    y, p, cells = (y[good], p[good], cells[good])
    unique, ix, count = np.unique(cells, return_inverse=True, return_counts=True)
    w = 1 / count[ix]
    mse = np.average((y - p) ** 2, weights=w)
    var = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    return [float(np.sqrt(mse)), float(1 - mse / var), len(unique)]
def metric(y, p):
    good = np.isfinite(p)
    err = np.mean((y[good] - p[good]) ** 2)
    return [float(np.sqrt(err)), float(1 - err / np.var(y[good])), int(good.sum())]
def display_xy(core, x, y):
    mask = x > 5 if core == 'HTTL2' else np.ones(len(x), bool)
    return (x[mask], y[mask])
def smooth(x, y, g):
    ok = np.isfinite(y)
    x = x[ok]
    y = y[ok]
    out = np.full(len(g), np.nan)
    if not len(x):
        return out
    b = np.floor(x / 0.25)
    ub = np.unique(b)
    xx = np.array([x[b == k].mean() for k in ub])
    yy = np.array([y[b == k].mean() for k in ub])
    dd = g[:, None] - xx
    w = np.exp(-0.5 * (dd / 0.2) ** 2) * (abs(dd) <= 0.6)
    den = w.sum(1)
    ok = (den > 0) & (g >= x.min()) & (g <= x.max())
    out[ok] = w[ok] @ yy / den[ok]
    return out
def run_rain(validation=None):
    from PIL import Image
    r = module(3)
    m = module(2)
    meta, views = r.load()
    lib = r.validated_plant_library()
    with Image.open(Path('D:/results/Supplementary Fig 2.png')) as im:
        reference = json.loads(im.info['Description'])
    groups = np.array([f'{int(np.floor(x / 5))}_{int(np.floor(z / 5))}' for x, z in meta[['Latitude', 'Longitude']].to_numpy()])
    fold = np.empty(len(meta), int)
    for h, (_, test) in enumerate(GroupKFold(5).split(meta, groups=groups)):
        fold[test] = h
    tests = [{} for _ in range(5)]
    tunes = [{} for _ in range(5)]
    tune_cells = [{} for _ in range(5)]
    allr = {}
    ff = {}
    cells = {}
    for c, v in views.items():
        a, f, y, rows = [v[k] for k in ['a', 'f', 'y', 'rows']]
        g = fold[rows]
        domain = cdist(a, f, 'cityblock').min(1) <= 1 + 2e-08
        idx = np.flatnonzero(domain)
        taxa = [j for j, t in enumerate(v['taxa']) if t in lib]
        plant, _ = r.botanical(a[idx][:, taxa], [lib[v['taxa'][j]] for j in taxa])
        common = np.zeros(len(y), bool)
        common[idx] = np.isfinite(np.column_stack(plant)).all(1)
        for h in range(5):
            print(f'MAP {c}: spatial outer fold {h+1}/5',flush=True)
            inner = (h + 1) % 5
            train = (g != h) & (g != inner)
            valid = common & (g == inner)
            hold = common & (g == h)
            tuned = r.raw_models(a[train], y[train], a[valid])
            lo = np.quantile(y[train & domain], 0.25)
            tune_cells[h][c] = meta.climate_cell.to_numpy()[rows[valid]]
            tunes[h][c] = (y[valid], np.column_stack(traditional(m, a[train], y[train], a[valid], rain=True) + [tuned]), y[valid] <= lo)
            train = g != h
            if validation is None:
                tested = r.raw_models(a[train], y[train], a[hold])
            else:
                test_rows, observations, predictions, fold_ids = validation[c]
                fold_mask = fold_ids == h
                np.testing.assert_array_equal(test_rows[fold_mask], rows[hold])
                np.testing.assert_array_equal(observations[fold_mask], y[hold])
                tested = predictions[fold_mask]
            tests[h][c] = (y[hold], np.column_stack(traditional(m, a[train], y[train], a[hold], rain=True) + [tested]), y[hold] <= np.quantile(y[train & domain], 0.25))
        allr[c] = tuple((np.concatenate(items) for items in zip(*(t[c] for t in tests))))
        cells[c] = np.concatenate([meta.climate_cell.to_numpy()[rows[common & (g == h)]] for h in range(5)])
        observed, pred, _ = allr[c]
        ref = reference['metrics'][c]
        np.testing.assert_allclose(np.sqrt(((pred[:, 3:] - observed[:, None]) ** 2).mean(0)), ref['rmse'], atol=1e-06)
        np.testing.assert_array_equal(v['age'], reference['age'][c])
        ff[c] = np.column_stack(traditional(m, a, y, f, rain=True) + [np.array(reference['predictions'][c])])
        print('Verified Fig. 3', c, flush=True)
    return finish(views, allr, ff, tests, tunes, cells, choose_rain, tune_cells=tune_cells)
def run(i):
    if i == 2:
        return run_rain()
    m = module(i)
    meta, V = m.load()
    groups = np.array([f'{int(np.floor(x / 5))}_{int(np.floor(z / 5))}' for x, z in meta[['Latitude', 'Longitude']].to_numpy()])
    fold = np.empty(len(meta), int)
    for h, (_, te) in enumerate(GroupKFold(5).split(meta, groups=groups)):
        fold[te] = h
    tests = [{} for _ in range(5)]
    tunes = [{} for _ in range(5)]
    tune_cells = [{} for _ in range(5)]
    allr = {}
    ff = {}
    cells = {}
    for c, v in V.items():
        a, f, y, rows = [v[k] for k in ['a', 'f', 'y', 'rows']]
        domain = cdist(a, f, 'cityblock').min(1) <= 1 + 2e-08
        g = fold[rows]
        oof = []
        for h in range(5):
            print(f'MAT {c}: spatial outer fold {h+1}/5',flush=True)
            hold = h
            val = (h + 1) % 5
            it = np.flatnonzero((g != hold) & (g != val))
            iv = np.flatnonzero((g == val) & domain)
            ie = np.flatnonzero((g == hold) & domain)
            z = fit(m, a[it], y[it], a[iv])
            tune_cells[h][c] = meta.climate_cell.to_numpy()[rows[iv]]
            tunes[h][c] = (y[iv], z)
            it = np.flatnonzero(g != hold)
            z = fit(m, a[it], y[it], a[ie])
            tests[h][c] = (y[ie], z)
            oof.append(tests[h][c])
        cells[c] = np.concatenate([meta.climate_cell.to_numpy()[rows[domain & (g == h)]] for h in range(5)])
        allr[c] = tuple((np.concatenate(z) for z in zip(*oof)))
        ff[c] = fit(m, a, y, f)
        print(i, c, flush=True)
    return finish(V, allr, ff, tests, tunes, cells, tune_cells=tune_cells)
def finish(V, allr, ff, tests, tunes, cells, selector=choose, tune_cells=None):
    return dict(variable='MAP' if selector is choose_rain else 'MAT',
                ages={c:v['age'] for c,v in V.items()},allr=allr,ff=ff,
                tests=tests,tunes=tunes,cells=cells,tune_cells=tune_cells)
def smooth_region(grid, trend, sigma):
    from scipy.ndimage import gaussian_filter1d
    supported = np.isfinite(trend)
    if not supported.any():
        return (grid[:0], trend[:0])
    segment = (grid >= grid[supported].min()) & (grid <= grid[supported].max())
    x = grid[segment]
    y = np.interp(x, grid[supported], trend[supported])
    return (x, gaussian_filter1d(y, sigma / (grid[1] - grid[0]), mode='nearest', truncate=3))
import argparse,gzip,hashlib
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
def input_fingerprint():
    import sklearn
    digest=hashlib.sha256(('weighted-calibration-v1;'+sklearn.__version__).encode())
    paths=[ROOT/f'Supplementary Fig {i-1}.py' for i in [2,3,5]]+[Path('D:/results/Supplementary Fig 2.png')]
    paths += [Path('D:/data/Processed data/relative_abundance/GALORE sift/GALORE_relative_abundance_sift.csv'),
              Path('D:/data/1. GALORE_locations.csv'),Path('D:/data/3. NLR.xlsx')]
    paths += sorted(Path('D:/data/Processed data/borehole').glob('*.csv'))
    paths += sorted(Path('D:/data/Processed data/Plant distribution').glob('*.csv'))
    for path in paths:
        digest.update(str(path).encode());digest.update(path.read_bytes())
    climate=Path('H:/Climatic data/month/climate14.nc')
    stat=climate.stat();digest.update(str((str(climate),stat.st_size,stat.st_mtime_ns)).encode())
    return digest.hexdigest()
BASE_INDEX=np.array([0,2,3,4,5])
GROUP_NAMES=['Modern Analogue','WA-PLS','ML']
MAIN='Equal groups'
CONTROL=['Without Modern Analogue','Without WA-PLS','Without ML']
DISPLAY={MAIN:'Equal groups (main)',**{n:n for n in CONTROL}}
FIXED_WEIGHTS={MAIN:np.full(3,1/3),
               'Without Modern Analogue':np.array([0,.5,.5]),
               'Without WA-PLS':np.array([.5,0,.5]),
               'Without ML':np.array([.5,.5,0])}
OMISSIONS=CONTROL
def group_predictions(base):
    p=np.asarray(base,float)[:,BASE_INDEX]
    ml=np.mean(p[:,2:5],axis=1)
    return np.column_stack([p[:,0],p[:,1],ml])
def combine(groups,weights):
    w=np.asarray(weights,float)
    assert len(w)==3 and (w>=0).all() and abs(w.sum()-1)<1e-9
    active=w>0
    return np.asarray(groups,float)[:,active]@w[active]
def evaluate(payload):
    var=payload['variable'];cores=list(payload['ages'])
    weights=FIXED_WEIGHTS
    fold_weights={name:[w]*5 for name,w in FIXED_WEIGHTS.items()}
    result=dict(methods=GROUP_NAMES,ML_definition='(RF + ExtraTrees + Boosting) / 3 in every test',cores={},schemes={})
    for core in cores:
        result['cores'][core]=dict(age=payload['ages'][core],group_predictions=group_predictions(payload['ff'][core]).tolist())
    for name in [MAIN]+CONTROL:
        model=dict(group_weights=weights[name].tolist(),outer_group_weights=[w.tolist() for w in fold_weights[name]],cores={})
        for core in cores:
            predictions=[];fold_scores=[];offset=0
            for h in range(5):
                values=payload['tests'][h][core]
                y=np.asarray(values[0],float)
                prediction=combine(group_predictions(values[1]),fold_weights[name][h])
                cells=np.asarray(payload['cells'][core])[offset:offset+len(y)];offset+=len(y)
                predictions.append(prediction);fold_scores.append(cell_metric(y,prediction,cells))
            prediction=np.concatenate(predictions);y=np.asarray(payload['allr'][core][0],float)
            assert len(y)==len(prediction)==len(payload['cells'][core])
            estimate=combine(result['cores'][core]['group_predictions'],weights[name])
            model['cores'][core]=dict(estimate=estimate.tolist(),
                cell_test=cell_metric(y,prediction,np.asarray(payload['cells'][core])),
                outer_test=metric(y,prediction),fold_cell_test=fold_scores)
        scores=np.array([v['cell_test'] for v in model['cores'].values()])
        model['macro_rmsep']=float(scores[:,0].mean());model['macro_r2']=float(scores[:,1].mean())
        result['schemes'][name]=model
        print(var,name,'weights:',weights[name].round(4),'macro RMSEP / R2:',round(model['macro_rmsep'],4),round(model['macro_r2'],4),flush=True)
    main_file=Path('D:/results/Fig. 2.png')
    if main_file.exists():
        main=load_cached(main_file)[var]
        for core,v in main['cores'].items():
            np.testing.assert_allclose(v['estimate'],result['schemes'][MAIN]['cores'][core]['estimate'],equal_nan=True)
            np.testing.assert_allclose(v['cell_test'],result['schemes'][MAIN]['cores'][core]['cell_test'])
    return result
def regional(result,name,region):
    grid=np.arange(0,18.0125,.025);smoothed=[]
    for core in REGIONS[region]:
        age=np.asarray(result['cores'][core]['age'],float)
        value=np.asarray(result['schemes'][name]['cores'][core]['estimate'],float)
        age,value=display_xy(core,age,value);keep=(age>=0)&(age<=18)
        smoothed.append(smooth(age[keep],value[keep],grid))
    smoothed=np.asarray(smoothed);count=np.isfinite(smoothed).sum(0)
    trend=np.divide(np.nansum(smoothed,axis=0),count,out=np.full(len(grid),np.nan),where=count>0)
    return smooth_region(grid,trend,.2 if region==0 else .4)
def display_core(core):
    return {'YH2':'YH','HTTL2':'HTTL'}.get(core,core)
def publication_style():
    plt.rcParams.update({'font.family':'Times New Roman','font.weight':'normal','font.size':22,
        'axes.labelsize':24,'axes.labelweight':'bold','axes.titlesize':20,'axes.titleweight':'bold',
        'xtick.labelsize':22,'ytick.labelsize':22,'legend.fontsize':20,
        'axes.linewidth':1.2,'axes.unicode_minus':True,'savefig.facecolor':'white'})
def publication_figure(rows,columns,table_rows,panel_height=2.4,legend_rows=3,header_lines=1):
    left,right=.085,.975;gap=.55;bottom=.28
    header_height=.56 if header_lines==1 else .80
    table_height=header_height+.44*table_rows
    legend_bottom=bottom+table_height+.70
    legend_top=legend_bottom+.25+.40*legend_rows
    plot_bottom=legend_top+.95
    plot_top=plot_bottom+rows*panel_height+(rows-1)*gap
    height=plot_top+.55
    fig=plt.figure(figsize=(16,height))
    grid=fig.add_gridspec(rows,columns,left=left,right=right,
        bottom=plot_bottom/height,top=plot_top/height,hspace=gap/panel_height,wspace=.20)
    axes=grid.subplots(sharex=True,sharey=False,squeeze=False)
    for row in range(rows):
        for ax in axes[row]:
            ax._publication_data=True
            if row<rows-1:ax.tick_params(axis='x',which='both',labelbottom=False)
    table_ax=fig.add_axes([left,bottom/height,right-left,table_height/height]);table_ax.axis('off')
    fig._publication_table_rows=table_rows
    fig._publication_header_height=header_height
    fig._publication_legend_top=legend_top/height
    return fig,axes,table_ax
def publication_legend(fig,handles,labels=None):
    kwargs=dict(loc='upper center',bbox_to_anchor=(.5,fig._publication_legend_top),
        ncols=3,frameon=False,fontsize=20,handletextpad=.45,columnspacing=1.2,
        borderaxespad=0,handlelength=2)
    legend=fig.legend(handles,labels,**kwargs) if labels is not None else fig.legend(handles=handles,**kwargs)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman');text.set_fontweight('normal')
    return legend
def publication_table(table,fig):
    table.auto_set_font_size(False);table.set_fontsize(20)
    header=fig._publication_header_height
    height=header+.44*fig._publication_table_rows
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#C6CDD1');cell.set_linewidth(.7)
        cell.set_facecolor('#EAF0F2' if row==0 else '#FFFFFF' if row%2 else '#F7F9FB')
        cell.set_height((header if row==0 else .44)/height)
        text=cell.get_text();text.set_fontfamily('Times New Roman');text.set_fontsize(20)
        text.set_fontweight('bold' if row==0 else 'normal')
def publication_finalize(fig):
    from matplotlib.ticker import MultipleLocator,MaxNLocator
    for ax in fig.axes:
        if getattr(ax,'_publication_data',False):
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_color('#3F4851');spine.set_linewidth(1.2)
            ax.grid(False,which='both');ax.grid(True,which='major',axis='both',color='#E2E6E9',lw=.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both',which='major',length=5,width=1.2,pad=5,labelsize=22)
            ax.tick_params(axis='both',which='minor',length=3,width=1)
            ax.xaxis.set_major_locator(MultipleLocator(2));ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.label.set_fontsize(24);ax.yaxis.label.set_fontsize(24)
            ax.xaxis.labelpad=12;ax.yaxis.labelpad=12
            for text in [ax.xaxis.label,ax.yaxis.label]:
                text.set_fontfamily('Times New Roman');text.set_fontweight('bold')
            for text in [*ax.get_xticklabels(),*ax.get_yticklabels()]:
                text.set_fontfamily('Times New Roman');text.set_fontweight('normal')
        for title in [ax.title,ax._left_title,ax._right_title]:
            title.set_fontfamily('Times New Roman');title.set_fontweight('bold');title.set_fontsize(20)
        for table in ax.tables:publication_table(table,fig)
def render(results):
    for result in results.values():
        result['schemes']={name:result['schemes'][name] for name in [MAIN]+CONTROL}
    publication_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    colors={MAIN:'#171717','Without Modern Analogue':'#D48025',
            'Without WA-PLS':'#8D5AA6','Without ML':'#258C75'}
    styles={MAIN:'-','Without Modern Analogue':'--','Without WA-PLS':':','Without ML':(0,(5,2))}
    fig, plot_axes, table_ax = publication_figure(2,2,4,panel_height=2.9,legend_rows=2,header_lines=2)
    for row,var in enumerate(['MAP','MAT']):
        axes=[];values=[]
        for region,region_name in enumerate(['West','East']):
            ax=plot_axes[row,region];axes.append(ax)
            curves={name:regional(results[var],name,region) for name in [MAIN]+CONTROL}
            age,base=curves[MAIN]
            for x,y in curves.values():np.testing.assert_allclose(x,age,atol=1e-12)
            omitted=np.array([curves[n][1] for n in OMISSIONS])
            ax.fill_between(age,omitted.min(0),omitted.max(0),color='#A8ADB3',alpha=.30,lw=0,zorder=1)
            for name in CONTROL:
                x,y=curves[name]
                ax.plot(x,y,color=colors[name],ls=styles[name],lw=2.2,zorder=2)
            ax.plot(age,base,color=colors[MAIN],lw=3.0,zorder=4)
            values.extend(y for x,y in curves.values())
            ax.set_title(f'({chr(97+2*row+region)}) {region_name}, Qaidam',loc='left',pad=10)
            ax.set_xlim(18,0);ax.grid(alpha=.14)
            if region==0:ax.set_ylabel('MAP (mm/year)' if var=='MAP' else 'MAT (°C)')
            if row==1:ax.set_xlabel('Age (Ma)')
            else:ax.tick_params(axis='x',labelbottom=False)
        values=np.concatenate(values);lo,hi=np.nanmin(values),np.nanmax(values);margin=.09*(hi-lo)
        for ax in axes:ax.set_ylim(lo-margin,hi+margin)
    handles=[Line2D([],[],color=colors[n],ls=styles[n],lw=2.5 if n==MAIN else 1.65,label=DISPLAY[n]) for n in [MAIN]+CONTROL]
    handles.append(Patch(facecolor='#A8ADB3',alpha=.30,label='Leave-one-method-out range'))
    publication_legend(fig,handles)
    ax=table_ax;ax.axis('off')
    ax.set_title('Climate-cell-weighted outer-test performance',pad=14)
    rows=[]
    for name in [MAIN]+CONTROL:
        map_model=results['MAP']['schemes'][name];mat_model=results['MAT']['schemes'][name]
        np.testing.assert_allclose(map_model['group_weights'],mat_model['group_weights'])
        weights=' / '.join(f'{w*100:.1f}' for w in map_model['group_weights'])
        rows.append([DISPLAY[name],weights,f"{map_model['macro_rmsep']:.2f}",f"{map_model['macro_r2']:.3f}",
                     f"{mat_model['macro_rmsep']:.3f}",f"{mat_model['macro_r2']:.3f}"])
    table=ax.table(cellText=rows,colLabels=['Scheme','Weights (%)\nModern Analogue / WA-PLS / ML',
        'MAP\nRMSEP','MAP R²','MAT\nRMSEP','MAT R²'],colWidths=[.23,.31,.13,.10,.13,.10],
        cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(13)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#C6CDD1');cell.set_linewidth(.6)
        if r==0:cell.set_facecolor('#EAF0F2')
    metadata=dict(results=results,analysis_role='Fixed three-method leave-one-out sensitivity',main_scheme=MAIN,
        group_definitions=GROUP_NAMES,ML_internal_weights={'RF':1/3,'ExtraTrees':1/3,'Boosting':1/3},
        shaded_range='Range of three leave-one-method-out curves; not a confidence interval',protocol=__doc__)
    output=Path('D:/results/Supplementary Fig 7.png')
    publication_finalize(fig)
    fig.savefig(output,dpi=300,metadata={'Description':json.dumps(serial(metadata))});plt.close(fig)
    print('Saved:',output,flush=True)
def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--refit',action='store_true')
    parser.add_argument('--training-cache',type=Path)
    a=parser.parse_args();ROOT=a.root
    output=Path('D:/results/Supplementary Fig 7.png')
    if output.exists() and not a.refit:
        with Image.open(output) as im:results=json.loads(im.info['Description'])['results']
        render(results);return
    signature=input_fingerprint() if a.training_cache else None
    results={}
    for var,index in [('MAP',2),('MAT',5)]:
        cached=a.training_cache/(var+'_calibration.json.gz') if a.training_cache else None
        if cached and cached.exists():
            with gzip.open(cached,'rt',encoding='utf-8') as f:container=json.load(f)
            if container['fingerprint']!=signature:raise ValueError('Calibration cache inputs changed; use a fresh cache directory.')
            payload=container['payload'];print('Reusing audited calibration arrays:',var,flush=True)
        else:
            payload=run(index)
            if cached:
                cached.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(cached,'wt',encoding='utf-8') as f:json.dump(serial(dict(fingerprint=signature,payload=payload)),f)
        results[var]=evaluate(payload)
    render(results)
if __name__=='__main__':main()
