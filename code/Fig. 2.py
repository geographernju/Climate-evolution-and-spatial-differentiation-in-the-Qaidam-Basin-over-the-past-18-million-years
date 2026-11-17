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
FIXED_METHODS = ['Modern Analogue', 'WA-PLS', 'RF', 'ExtraTrees', 'Boosting']
COMB = [tuple(NAMES.index(method) for method in FIXED_METHODS)]
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
    return finish(views, allr, ff, tests, tunes, cells, choose_rain)
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
            tunes[h][c] = (y[iv], z)
            it = np.flatnonzero(g != hold)
            z = fit(m, a[it], y[it], a[ie])
            tests[h][c] = (y[ie], z)
            oof.append(tests[h][c])
        cells[c] = np.concatenate([meta.climate_cell.to_numpy()[rows[domain & (g == h)]] for h in range(5)])
        allr[c] = tuple((np.concatenate(z) for z in zip(*oof)))
        ff[c] = fit(m, a, y, f)
        print(i, c, flush=True)
    return finish(V, allr, ff, tests, tunes, cells)
def finish(V, allr, ff, tests, tunes, cells, selector=choose):
    methods=['Modern Analogue','WA-PLS','RF','ExtraTrees','Boosting']
    index=[NAMES.index(method) for method in methods]
    weights=np.array([1/3,1/3,1/9,1/9,1/9])
    cores={}
    for core,v in V.items():
        y=np.asarray(allr[core][0],float)
        pred=np.concatenate([np.asarray(tests[h][core][1])[:,index]@weights for h in range(5)])
        members=np.asarray(ff[core])[:,index]
        cores[core]=dict(age=np.asarray(v['age']).tolist(),estimate=(members@weights).tolist(),
            member_predictions={method:members[:,i].tolist() for i,method in enumerate(methods)},
            outer_test=metric(y,pred),cell_test=cell_metric(y,pred,cells[core]))
    return dict(methods=methods,weights=weights.tolist(),outer_weights=[weights.tolist()]*5,
        outer_selections=[methods]*5,groups={'Modern Analogue':1/3,'WA-PLS':1/3,'ML':1/3},cores=cores)
def smooth_region(grid, trend, sigma):
    from scipy.ndimage import gaussian_filter1d
    supported = np.isfinite(trend)
    if not supported.any():
        return (grid[:0], trend[:0])
    segment = (grid >= grid[supported].min()) & (grid <= grid[supported].max())
    x = grid[segment]
    y = np.interp(x, grid[supported], trend[supported])
    return (x, gaussian_filter1d(y, sigma / (grid[1] - grid[0]), mode='nearest', truncate=3))
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
def main(saved=None):
    publication_style()
    from matplotlib.lines import Line2D
    markers = {'Modern Analogue': 'P', 'WA': 'v', 'WA-PLS': 'D', 'RF': 'o', 'ExtraTrees': '^', 'Boosting': 's'}
    labels = {'RF': 'RF', 'ExtraTrees': 'ExtraTrees', 'Boosting': 'Boosting'}
    fig, plot_axes, tab = publication_figure(2,1,2,panel_height=3.2,legend_rows=3)
    axes=plot_axes[:,0]
    results = {}
    for j, (i, label) in enumerate([(2, 'MAP'), (5, 'MAT')]):
        d = run(i) if saved is None else saved[label]
        results[label] = d
        print(label, d['methods'], {c: v['outer_test'] for c, v in d['cores'].items()}, flush=True)
        display_label = 'MAT (°C)' if i == 5 else 'MAP (mm/year)'
        g = np.arange(0, 18.0125, 0.025)
        ax = axes[j]
        trend_values = []
        for cs, name, color in zip(REGIONS, ['West', 'East'], ['#24678c', '#b24d35']):
            curves = []
            series = []
            for core in cs:
                v = d['cores'][core]
                age, estimate = display_xy(core, np.asarray(v['age']), np.asarray(v['estimate']))
                keep = (age >= 0) & (age <= 18.0)
                for method in d['methods']:
                    _, member = display_xy(core, np.asarray(v['age']), np.asarray(v['member_predictions'][method]))
                    ax.scatter(age[keep], member[keep], s=13, marker=markers[method], facecolors='none', edgecolors=color, linewidths=0.55, alpha=0.4, zorder=2)
                series.append((age[keep], estimate[keep]))
                curves.append(smooth(age[keep], estimate[keep], g))
            curves = np.asarray(curves)
            n = np.isfinite(curves).sum(0)
            trend = np.divide(np.nansum(curves, axis=0), n, out=np.full_like(g, np.nan), where=n > 0)
            trend_age, trend_value = smooth_region(g, trend, 0.2 if name == 'West' else 0.4)
            low_band, high_band = trend_band(series, g, trend_age, trend_value, 0.2, 0.2 if name == 'West' else 0.4)
            ax.fill_between(trend_age, low_band, high_band, color=color, alpha=0.24, linewidth=0, zorder=1)
            trend_values.extend([trend_value, low_band, high_band])
            ax.plot(trend_age, trend_value, color=color, ls='-', lw=3.0, zorder=4)
        values = np.concatenate(trend_values)
        low, high = (np.nanmin(values), np.nanmax(values))
        margin = 0.2 * (high - low) if high > low else max(abs(low) * 0.05, 1.0)
        lower, upper = (low - margin, high + margin)
        ax.set_ylim(lower, upper)
        ax.set_xlim(18.0, 0)
        ax.set_ylabel(display_label)
        ax.grid(alpha=0.14)
        ax.set_title('('+chr(97+j)+')',loc='left',pad=10)
        handles = [Line2D([], [], ls='none', marker=markers[m], markerfacecolor='none', markeredgecolor='#373737', markersize=6, label=labels.get(m,m)) for m in d['methods']]
    axes[-1].set_xlabel('Age (Ma)')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, lw=2.7, label=name+', Qaidam') for name, color in zip(['West', 'East'], ['#24678c', '#b24d35'])]
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor='#84919a', alpha=0.36, label=BAND_LABEL))
    handles.extend([Line2D([],[],ls='none',marker=markers[m],markerfacecolor='none',markeredgecolor='#373737',markersize=7,label=m) for m in results['MAP']['methods']])
    publication_legend(fig,handles)
    tab.axis('off')
    tab.set_title('Climate-cell-weighted RMSEP / predictive R²',pad=14)
    cores = sum(REGIONS, [])
    rows = []
    for label, d in results.items():
        row = ['MAT (°C)' if label.startswith('MAT') else 'MAP (mm/year)']
        for core in cores:
            rmsep, r2, _ = d['cores'][core]['cell_test']
            row.append(f'{rmsep:.1f} / {r2:.2f}')
        rows.append(row)
    table = tab.table(cellText=rows, colLabels=['Climate variable', 'KC-1', 'SG-1b', 'SG-1', 'DHG', 'YH', 'HTTL'], colWidths=[.22]+[.13]*6, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#edf1f2')
    results['protocol'] = 'Main reconstruction: fixed equal weighting of three method groups: Modern Analogue 1/3, WA-PLS 1/3 and ML 1/3. The ML group is the equal mean of RF, ExtraTrees and Boosting; each ML algorithm therefore receives 1/9 of the final estimate. This is category-balanced weighting, not weighting optimized for minimum validation error. All methods share pollen and calibration inputs. The five methods and coefficients remain fixed in every spatial outer fold. Earlier screening and benchmark comparisons informed this choice; validation is exploratory, not a newly independent test. West: KC-1, SG-1b, SG-1; East: DHG, YH, HTTL. Display 0-18 Ma, HTTL >5 Ma. Scatter points are the original member predictions. Within-core 0.25 Ma bin means use Gaussian sigma 0.20 Ma, cutoff 0.60 Ma. Regional curves equally average contributing cores, interpolate internal gaps, then use Gaussian sigma 0.20 Ma West and 0.40 Ma East; no extrapolation beyond support. Metrics are spatial outer-test RMSEP and predictive R2, available by sample and climate-cell weighting. Shading is the occupied-bin resampling band with fitted models and coefficients fixed, not full reconstruction uncertainty.'
    publication_finalize(fig)
    fig.savefig(Path('D:/results/Fig. 2.png'), dpi=300, metadata={'Description': json.dumps(add_definitions(dict(results, uncertainty=BAND_NOTE)))})
if __name__=='__main__':
    import sys
    if '--refit' in sys.argv or not Path('D:/results/Fig. 2.png').exists():
        main()
    else:
        main(saved=load_cached(Path('D:/results/Fig. 2.png')))
