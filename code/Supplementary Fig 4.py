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
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import json, hashlib, tempfile
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
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
ROOT = Path(__file__).resolve().parent
CORES = {'KC-1': 'KC-1.csv', 'SG-1b': 'SG-1b.csv', 'SG-1': 'SG-1.csv', 'DHG': 'DHG.csv', 'YH2': 'YH.csv', 'HTTL2': 'HTTL.csv'}
SEED = 20261027
def norm(a):
    return a / np.maximum(a.sum(1, keepdims=True), 1e-20)
def scores(y, p):
    e = p - y
    return dict(n=len(y), RMSEP=float(np.sqrt(np.mean(e * e))), r2=float(np.corrcoef(y, p)[0, 1] ** 2), R2pred=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)), bias=float(e.mean()))
def climate():
    from netCDF4 import Dataset
    path = Path('H:/Climatic data/month/climate14.nc')
    st = path.stat()
    key = hashlib.sha256(str((str(path), st.st_size, st.st_mtime_ns, 'tmp-1990-2019-monthly-mean-v1')).encode()).hexdigest()
    cache = Path(tempfile.gettempdir()) / ('climate_' + key + '.npz')
    if cache.exists():
        z = np.load(cache)
        return (z['field'], z['lat'], z['lon'])
    with Dataset(path) as d:
        assert d['tmp'].units == 'degrees Celsius'
        t = np.array(d['time'][:])
        lat = np.array(d['latitude'][:])
        lon = np.array(d['longitude'][:])
        total = np.zeros((len(lat), len(lon)))
        valid = np.ones(total.shape, bool)
        for year in range(1990, 2020):
            ix = [np.flatnonzero(t == year * 100 + m) for m in range(1, 13)]
            assert all((len(i) == 1 for i in ix))
            v = np.ma.filled(d['tmp'][np.concatenate(ix)], np.nan).astype(float)
            good = np.isfinite(v).all(0)
            total += np.where(good, v.mean(0), 0)
            valid &= good
    field = np.where(valid, total / 30, np.nan)
    np.savez_compressed(cache, field=field, lat=lat, lon=lon)
    return (field, lat, lon)
def near(axis, x):
    k = np.clip(np.searchsorted(axis, x), 1, len(axis) - 1)
    return np.where(abs(x - axis[k - 1]) <= abs(x - axis[k]), k - 1, k)
def load():
    import unicodedata
    def key(s):
        return ' '.join(unicodedata.normalize('NFC', str(s)).strip().split()).casefold()
    field, lat, lon = climate()
    raw = pd.read_csv('D:/data/Processed data/relative_abundance/GALORE sift/GALORE_relative_abundance_sift.csv').fillna(0)
    raw.columns = raw.columns.str.strip()
    raw['key'] = raw.Name.map(key)
    loc = pd.read_csv('D:/data/1. GALORE_locations.csv', encoding='gb18030', low_memory=False)
    loc.columns = loc.columns.str.strip()
    loc['key'] = loc.Name.map(key)
    for c in ['Longitude', 'Latitude']:
        loc[c] = pd.to_numeric(loc[c], errors='coerce')
    loc = loc[loc.Longitude.between(-180, 180) & loc.Latitude.between(-90, 90)].copy()
    loc['pair'] = list(zip(loc.Longitude.round(4), loc.Latitude.round(4)))
    counts = loc.groupby('key').pair.nunique()
    loc = loc[~loc.key.isin(counts[counts > 1].index)].drop_duplicates('key')
    meta = raw.merge(loc[['key', 'Longitude', 'Latitude']], on='key', how='inner', validate='many_to_one')
    iy = near(lat, meta.Latitude.to_numpy())
    ix = near(lon, meta.Longitude.to_numpy())
    meta['climate_cell'] = iy * len(lon) + ix
    meta['MAT'] = field[iy, ix]
    meta = meta[np.isfinite(meta.MAT)].reset_index(drop=True)
    meta['SiteLatitude'] = meta.Latitude
    meta['SiteLongitude'] = meta.Longitude
    meta['Latitude'] = lat[meta.climate_cell.to_numpy() // len(lon)]
    meta['Longitude'] = lon[meta.climate_cell.to_numpy() % len(lon)]
    mapping = pd.read_excel('D:/data/3. NLR.xlsx').dropna(subset=['Standardized pollen taxon', 'GALORE comparable taxon'])
    pairs = mapping[['Standardized pollen taxon', 'GALORE comparable taxon']].astype(str).apply(lambda s: s.str.strip()).drop_duplicates()
    assert not pairs.iloc[:, 0].duplicated().any()
    mp = dict(pairs.values)
    views = {}
    meta['Picea+Abies'] = meta['Picea'] + meta['Abies']
    for core, fn in CORES.items():
        f = pd.read_csv(Path('D:/data/Processed data/borehole') / fn).fillna(0)
        cols = [c for c in f if c != 'Age (Ma)']
        group = {}
        for t in cols:
            dst = mp.get(t, t)
            if core == 'YH2' and t == 'Picea':
                if 'Abies' in cols:
                    raise ValueError('Review separately reported YH Abies')
                dst = 'Picea+Abies'
            if dst in meta:
                group.setdefault(dst, []).append(t)
        taxa = sorted(group)
        a = meta[taxa].to_numpy(float)
        fa = np.column_stack([f[group[t]].sum(1) for t in taxa])
        rows = np.flatnonzero(a.sum(1) > 0)
        coverage = fa.sum(1) / f[cols].sum(1).to_numpy()
        assert np.allclose(coverage, 1, atol=1e-05)
        assert (a >= 0).all() and (fa >= 0).all()
        views[core] = dict(a=norm(a[rows]), f=norm(fa), rows=rows, y=meta.MAT.to_numpy()[rows], age=f['Age (Ma)'].to_numpy(), taxa=taxa, coverage=coverage)
    return (meta, views)
def splice(views, recon, cores):
    cuts = []
    for older, younger in zip(cores[:-1], cores[1:]):
        low = max(views[older]['age'].min(), views[younger]['age'].min())
        high = min(views[older]['age'].max(), views[younger]['age'].max())
        assert low <= high
        cuts.append(float((low + high) / 2))
    chunks = []
    for j, core in enumerate(cores):
        age = views[core]['age']
        mask = np.ones(len(age), bool)
        if j > 0:
            mask &= age <= cuts[j - 1]
        if j < len(cuts):
            mask &= age > cuts[j]
        for x, y in zip(age[mask], recon[core][mask]):
            chunks.append((float(x), float(y), core))
    return (sorted(chunks, reverse=True), cuts)
CANDS = []
for k in [5, 20, 50]:
    CANDS.append(('Modern Analogue', k))
for k in [100, 300]:
    for alpha in [0.01, 0.1]:
        CANDS.append(('LocalRidge', k, alpha))
for transform in ['raw', 'sqrt']:
    for k in [1, 2, 3]:
        CANDS.append(('WA-PLS', transform, k))
for family in ['RF', 'ExtraTrees']:
    for mode, features, leaf in [('global', 2, 1), ('global', 0.5, 5), ('analogue', 0.5, 5)]:
        CANDS.append((family, mode, features, leaf))
for mode in ['global', 'analogue']:
    CANDS.append(('Boosting', mode))
for weight in [0.25, 0.5]:
    CANDS.append(('Hybrid', weight))
FAMILIES = list(dict.fromkeys((c[0] for c in CANDS)))
def wa(a, y, q, components):
    C = a.sum(0)
    keep = C > 1e-12
    a = a[:, keep]
    q = q[:, keep]
    C = C[keep]
    mu = y.mean()
    H = a.T @ a
    res = a.T @ (y - mu)
    g = res / C
    direction = g.copy()
    gamma = res @ g
    b = np.zeros(len(C))
    out = []
    for k in range(components):
        hd = H @ direction
        den = direction @ hd
        if den > 1e-18:
            step = gamma / den
            b += step * direction
            res -= step * hd
            g = res / C
            new = res @ g
            direction = g + new / max(gamma, 1e-30) * direction
            gamma = new
        out.append(mu + q @ b / np.maximum(q.sum(1), 1e-20))
    return np.column_stack(out)
METHOD = 'Modern Analogue'
ACTIVE = np.array([i for i, c in enumerate(CANDS) if c[0] == METHOD])
def predict(a, y, q, domain):
    result = np.zeros((len(q), len(CANDS)))
    root = np.sqrt(a)
    qr = np.sqrt(q)
    if METHOD in ['Modern Analogue', 'LocalRidge', 'Hybrid']:
        for start in range(0, len(q), 150):
            stop = min(len(q), start + 150)
            dist = cdist(qr[start:stop], root, 'sqeuclidean')
            br = 0.5 * cdist(q[start:stop], a, 'cityblock')
            limited = dist.copy()
            limited[br > 0.5 + 1e-08] = np.inf
            empty = ~np.isfinite(limited).any(1)
            limited[empty] = dist[empty]
            kk = min(300, len(a))
            ix = np.argpartition(limited, kk - 1, axis=1)[:, :kk]
            dd = np.take_along_axis(limited, ix, 1)
            order = np.argsort(dd, axis=1)
            ix = np.take_along_axis(ix, order, 1)
            dd = np.take_along_axis(dd, order, 1)
            for j, c in enumerate(CANDS):
                if c[0] == 'Modern Analogue':
                    k = c[1]
                    ds = dd[:, :k]
                    ii = ix[:, :k]
                    w = 1 / np.maximum(ds, 1e-12)
                    zero = ds <= 1e-12
                    has = zero.any(1)
                    w[has] *= zero[has]
                    result[start:stop, j] = (w * y[ii]).sum(1) / w.sum(1)
            for i in range(stop - start) if METHOD == 'LocalRidge' else []:
                for k in [100, 300]:
                    valid = np.isfinite(dd[i, :k])
                    ii = ix[i, :k][valid]
                    ds = dd[i, :k][valid]
                    if len(ii) < 10:
                        ii = np.argpartition(dist[i], min(k, len(a)) - 1)[:min(k, len(a))]
                        ds = dist[i, ii]
                    w = np.exp(-ds / max(np.median(ds), 1e-06))
                    w /= w.sum()
                    x = root[ii]
                    ym = w @ y[ii]
                    xm = w @ x
                    xc = x - xm
                    yc = y[ii] - ym
                    h = xc.T @ (w[:, None] * xc)
                    rhs = xc.T @ (w * yc)
                    scale = max(np.trace(h) / len(h), 1e-08)
                    for alpha in [0.01, 0.1]:
                        coef = solve(h + np.eye(len(h)) * alpha * scale, rhs, assume_a='pos', check_finite=False)
                        value = max(0, ym + (qr[start + i] - xm) @ coef)
                        result[start + i, CANDS.index(('LocalRidge', k, alpha))] = value
    for transform in ['raw', 'sqrt'] if METHOD == 'WA-PLS' else []:
        aa = norm(root) if transform == 'sqrt' else a
        qq = norm(qr) if transform == 'sqrt' else q
        pr = wa(aa, y, qq, 3)
        for k in [1, 2, 3]:
            result[:, CANDS.index(('WA-PLS', transform, k))] = pr[:, k - 1]
    for j, c in enumerate(CANDS):
        if not (c[0] == METHOD and METHOD in ['RF', 'ExtraTrees', 'Boosting'] or (METHOD == 'Hybrid' and c == ('RF', 'global', 0.5, 5))):
            continue
        tr = np.flatnonzero(domain) if c[1] == 'analogue' and np.sum(domain) >= 30 else np.arange(len(a))
        if c[0] == 'Boosting':
            m = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=10, learning_rate=0.05, early_stopping=False, random_state=SEED)
        else:
            cls = RandomForestRegressor if c[0] == 'RF' else ExtraTreesRegressor
            m = cls(n_estimators=500, max_features=c[2], min_samples_leaf=c[3], n_jobs=4, random_state=SEED)
        m.fit(a[tr], y[tr])
        result[:, j] = m.predict(q)
    for weight in [0.25, 0.5] if METHOD == 'Hybrid' else []:
        result[:, CANDS.index(('Hybrid', weight))] = weight * result[:, CANDS.index(('Modern Analogue', 20))] + (1 - weight) * result[:, CANDS.index(('RF', 'global', 0.5, 5))]
    assert np.isfinite(result).all()
    return result[:, ACTIVE]
def standalone_wa(a, y, q):
    mass = a.sum(0)
    opt = np.divide(a.T @ y, mass, out=np.zeros_like(mass), where=mass > 0)
    def raw(x):
        w = x[:, mass > 0]
        return w @ opt[mass > 0] / np.where(w.sum(1) > 0, w.sum(1), np.nan)
    train = raw(a)
    b = np.polyfit(train, y, 1)
    return np.polyval(b, raw(q))
names = ['Modern Analogue', 'WA', 'WA-PLS']
settings = {'Modern Analogue': ('Modern Analogue', 5), 'WA-PLS': ('WA-PLS', 'sqrt', 2)}
out = Path('D:/results/Supplementary Fig 4.png')
protocol = '1990-2019 tmp: mean of twelve monthly means then mean of annual means. Fixed hyperparameters; models refitted to temperature in five spatial folds and on the full training set. No precipitation parameter selection.'
def nearest(axis, v):
    k = np.clip(np.searchsorted(axis, v), 1, len(axis) - 1)
    return np.where(abs(v - axis[k - 1]) <= abs(v - axis[k]), k - 1, k)
def sample_cells(lons, lats, climate, lat, lon):
    lons = (np.asarray(lons) + 180) % 360 - 180
    lats = np.asarray(lats)
    iy, ix = (nearest(lat, lats), nearest(lon, lons))
    good = (abs(lats - lat[iy]) <= np.median(np.diff(lat)) / 2 + 1e-06) & (abs(lons - lon[ix]) <= np.median(np.diff(lon)) / 2 + 1e-06)
    values = climate[iy, ix].copy()
    values[~good] = np.nan
    return (values, iy * len(lon) + ix)
def read_taxon(path, climate, lat, lon, use_counts):
    d = pd.read_csv(path, encoding='utf-8-sig', sep=None, engine='python')
    d.columns = d.columns.str.strip()
    cols = ['longitude', 'latitude', 'occurrence_count']
    if not set(cols) <= set(d.columns):
        raise ValueError(f'{path.name}: missing columns')
    d = d[cols].apply(pd.to_numeric, errors='coerce')
    d = d[np.isfinite(d).all(axis=1) & d.latitude.between(-90, 90) & d.longitude.between(-180, 360) & (d.occurrence_count > 0)].copy()
    d['rain'], d['cell'] = sample_cells(d.longitude.values, d.latitude.values, climate, lat, lon)
    d = d[np.isfinite(d.rain)]
    d = d.groupby('cell', as_index=False).agg(longitude=('longitude', 'first'), latitude=('latitude', 'first'), rain=('rain', 'first'), occurrence_count=('occurrence_count', 'sum'))
    if len(d) < 10 or d.rain.std() == 0:
        raise ValueError(f'{path.stem}: fewer than 10 valid cells or constant rainfall')
    w = d.occurrence_count.values.astype(float) if use_counts else np.ones(len(d))
    return (d, w / w.sum())
def botanical(q, r):
    z = q > 0
    ok = z.any(1)
    mn = np.array([x.min() for x in r])
    mx = np.array([x.max() for x in r])
    mu = np.array([x.mean() for x in r])
    var = np.array([x.var(ddof=1) for x in r])
    lo = np.where(z, mn, -np.inf).max(1)
    hi = np.where(z, mx, np.inf).min(1)
    valid = ok & (lo <= hi)
    ca = np.full(len(q), np.nan)
    ca[valid] = (lo[valid] + hi[valid]) / 2
    sdinv = z @ (1 / var)
    pc = np.divide(z @ (mu / var), sdinv, out=np.full(len(z), np.nan), where=ok)
    grid = np.arange(np.floor(mn.min()), mx.max() + 0.1, 0.1)
    scores = []
    for x in r:
        f = np.searchsorted(x, grid, side='right') / len(x)
        scores.append(np.select([(f <= 0) | (f >= 1), (f <= 0.01) | (f > 0.99), (f <= 0.05) | (f > 0.95), (f <= 0.1) | (f > 0.9)], [0, 1, 2, 3], 4))
    scores = np.array(scores, dtype=float)
    ml = np.full(len(z), np.nan)
    mh = ml.copy()
    for i in range(0, len(z), 128):
        zz = z[i:i + 128]
        v = zz @ scores
        best = v.max(1)
        valid = (best >= zz.sum(1)) & zz.any(1)
        ml[i:i + 128] = np.where(valid, grid[v.argmax(1)], np.nan)
        mh[i:i + 128] = np.where(valid, grid[-1 - v[:, ::-1].argmax(1)], np.nan)
    cg = grid
    pdf = []
    for x in r:
        w = 1 / bg[np.minimum(np.floor(x - offset).astype(int), len(bg) - 1)].clip(1)
        w /= w.sum()
        lx = x
        mu = w @ lx
        sd = np.sqrt(w @ (lx - mu) ** 2)
        pdf.append(-0.5 * ((cg - mu) / max(sd, 0.01)) ** 2)
    crest = np.full(len(q), np.nan)
    for i in range(0, len(q), 128):
        a = q[i:i + 128]
        has = a.sum(1) > 0
        crest[i:i + 128] = np.where(has, cg[np.argmax(norm(a) @ np.array(pdf), axis=1)], np.nan)
    return ([ca, (ml + mh) / 2, pc, crest], [(np.where(np.isfinite(ca), lo, np.nan), np.where(np.isfinite(ca), hi, np.nan)), (ml, mh)])
def validated_plant_library():
    global bg, offset
    field, lat, lon = climate()
    lib = {}
    offset = int(np.floor(np.nanmin(field)))
    bg = np.bincount(np.floor(field[np.isfinite(field)] - offset).astype(int))
    joint_cells = {}
    for p in sorted(Path('D:/data/Processed data/Plant distribution').glob('*.csv')):
        d, _ = read_taxon(p, field, lat, lon, False)
        lib[p.stem] = np.sort(d.rain.to_numpy())
        if p.stem in ('Picea', 'Abies'):
            joint_cells[p.stem] = d.cell.to_numpy()
    merged_cells = np.union1d(joint_cells['Picea'], joint_cells['Abies'])
    lib['Picea+Abies'] = np.sort(field.ravel()[merged_cells])
    return lib
def main(validation=None):
    lib = validated_plant_library()
    global METHOD, ACTIVE
    meta, views = load()
    groups = np.array([f'{int(np.floor(a / 5))}_{int(np.floor(b / 5))}' for a, b in meta[['Latitude', 'Longitude']].to_numpy()])
    folds = list(GroupKFold(5).split(meta, groups=groups))
    data = {}
    stats = {}
    ranges = {}
    for c, v in views.items():
        a, f, rows, y = [v[k] for k in ['a', 'f', 'rows', 'y']]
        dom = cdist(a, f, 'cityblock').min(1) <= 1 + 2e-08
        P = np.full((len(y), len(names)), np.nan)
        F = []
        for j, METHOD in enumerate(names):
            ACTIVE = np.array([i for i, p in enumerate(CANDS) if p[0] == METHOD])
            def fit(tr, q, index):
                if METHOD == 'WA':
                    return standalone_wa(a[tr], y[tr], q)
                cfg = settings[METHOD]
                return predict(a[tr], y[tr], q, dom[tr])[:, list(ACTIVE).index(CANDS.index(cfg))]
            for n, (tr, te) in enumerate(folds):
                t = np.flatnonzero(np.isin(rows, tr))
                e = np.flatnonzero(np.isin(rows, te) & dom)
                P[e, j] = fit(t, a[e], n + 1)
            F.append(fit(np.arange(len(y)), f, 0))
        plant_rows = np.flatnonzero(dom)
        ix = [k for k, t in enumerate(v['taxa']) if t in lib]
        rain = [lib[v['taxa'][k]] for k in ix]
        pt, _ = botanical(a[plant_rows][:, ix], rain)
        plant_ok = np.zeros(len(y), bool)
        plant_ok[plant_rows] = np.isfinite(np.column_stack(pt)).all(1)
        mask = dom & plant_ok & np.isfinite(P).all(1)
        yy = y[mask]
        E = P[mask] - yy[:, None]
        stats[c] = dict(n=len(yy), rmse=np.sqrt((E ** 2).mean(0)).tolist(), r2=(1 - (E ** 2).sum(0) / ((yy - yy.mean()) ** 2).sum()).tolist())
        data[c] = np.column_stack(F)
        selected_rows = rows[mask]
        fold_id = np.full(len(meta), -1, int)
        cells = meta.climate_cell.to_numpy()[selected_rows]
        unique, index, counts = np.unique(cells, return_inverse=True, return_counts=True)
        weights = 1 / counts[index]
        mse = np.average((P[mask] - yy[:, None]) ** 2, axis=0, weights=weights)
        variance = np.average((yy - np.average(yy, weights=weights)) ** 2, weights=weights)
        stats[c].update(cell_rmsep=np.sqrt(mse).tolist(), cell_r2=(1 - mse / variance).tolist(), n_cells=len(unique))
        for h, (_, te) in enumerate(folds):
            fold_id[te] = h
        assert np.isfinite(yy).all() and len(np.unique(selected_rows)) == len(yy)
        stats[c]['validation_signature'] = hashlib.sha256(selected_rows.astype('<i8').tobytes() + yy.astype('<f8').tobytes() + fold_id[selected_rows].astype('<i8').tobytes()).hexdigest()
        if validation is not None:
            validation[c] = (selected_rows, yy, P[mask], fold_id[selected_rows])
    render(views, data, stats, ranges)
def smooth_region(grid, trend):
    from scipy.ndimage import gaussian_filter1d
    supported = np.isfinite(trend)
    if not supported.any():
        return (grid[:0], trend[:0])
    segment = (grid >= grid[supported].min()) & (grid <= grid[supported].max())
    x = grid[segment]
    y = np.interp(x, grid[supported], trend[supported])
    return (x, gaussian_filter1d(y, 0.4 / (grid[1] - grid[0]), mode='nearest', truncate=3))
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
def render(views, data, stats, ranges):
    publication_style()
    oldest = 18.0
    shown = {}
    for core, v in views.items():
        age = np.asarray(v['age'])
        keep = (age >= 0) & (age <= oldest)
        if core == 'HTTL2':
            keep &= age > 5
        shown[core] = np.flatnonzero(keep)[np.argsort(age[keep], kind='stable')]
    fig, axes, tab = publication_figure(len(names),2,len(names),panel_height=2.4,legend_rows=3)
    regions = [['KC-1', 'SG-1b', 'SG-1'], ['DHG', 'YH2', 'HTTL2']]
    color = ['#217c84', '#9a593c', '#66559c', '#47704a']
    for j, method in enumerate(names):
        for k, cs in enumerate(regions):
            ax = axes[j, k]
            palette = {'KC-1': '#804592', 'SG-1b': '#259981', 'SG-1': '#2678ad', 'HTTL2': '#be761e', 'YH2': '#c44e67', 'DHG': '#386641'}
            grid = np.arange(0, oldest + 0.0125, 0.025)
            sm = []
            series = []
            for c in cs:
                age = views[c]['age']
                ix = shown[c]
                x = age[ix]
                y = data[c][ix, j]
                series.append((x, y))
                valid_points = np.isfinite(y)
                bin_id = np.floor(x[valid_points] / 0.25).astype(int)
                bin_age = x[valid_points]
                bin_value = y[valid_points]
                groups = np.unique(bin_id)
                ax.scatter([np.median(bin_age[bin_id == h]) for h in groups], [np.median(bin_value[bin_id == h]) for h in groups], s=26, marker={'KC-1': 'o', 'SG-1b': '^', 'SG-1': 's', 'DHG': 'o', 'YH2': '^', 'HTTL2': 's'}[c], facecolors='none', edgecolors=palette[c], linewidths=1.1, alpha=0.82, label=display_core(c), zorder=3)
                if method in ['CA', 'MCR-w']:
                    lo, hi = ranges[c][names.index(method) - names.index('CA')]
                    ax.fill_between(x, lo[ix], hi[ix], color=palette[c], alpha=0.11, linewidth=0, label='Reconstructed range')
                good = np.isfinite(x) & np.isfinite(y)
                xx = x[good]
                yy = y[good]
                z = np.full(len(grid), np.nan)
                if len(xx):
                    bins = np.floor(xx / 0.25).astype(int)
                    b = np.unique(bins)
                    bx = np.array([xx[bins == h].mean() for h in b])
                    by = np.array([yy[bins == h].mean() for h in b])
                    dist = grid[:, None] - bx
                    w = np.exp(-0.5 * (dist / 0.6) ** 2) * (abs(dist) <= 1.8)
                    den = w.sum(1)
                    valid = (den > 0) & (grid >= xx.min()) & (grid <= xx.max())
                    z[valid] = w[valid] @ by / den[valid]
                sm.append(z)
            sm = np.array(sm)
            count = np.isfinite(sm).sum(0)
            trend = np.divide(np.nansum(sm, 0), count, out=np.full(len(grid), np.nan), where=count > 0)
            trend_age, trend_value = smooth_region(grid, trend)
            low_band, high_band = trend_band(series, grid, trend_age, trend_value, 0.6, 0.4)
            ax.fill_between(trend_age, low_band, high_band, color='#536979', alpha=0.28, linewidth=0, label=BAND_LABEL, zorder=2)
            ax.plot(trend_age, trend_value, color='#171717', ls='--', lw=2.8, zorder=4, label='Smoothed regional trend')
            ax.set_xlim(oldest, 0)
            ax.grid(axis='y', color='#e0e5e7', lw=0.65)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            for side in ['bottom', 'left']:
                ax.spines[side].set_color('#899499')
            style = dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1)
            ax.set_title(f'({chr(97+2*j+k)}) '+['West, Qaidam','East, Qaidam'][k],loc='left',pad=10)
            ax.set_title(method,loc='right',pad=10)
            if k==0:ax.set_ylabel('MAT (°C)')
            if j == len(names) - 1:
                ax.set_xlabel('Age (Ma)')
            low, high = ax.dataLim.intervaly
            if np.isfinite([low, high]).all():
                padding = 0.04 * (high - low) if high > low else max(0.04 * abs(low), 1.0)
                ax.set_ylim(low - padding, high + padding)
    handles = {}
    for panel in axes.flat:
        artists, labels = panel.get_legend_handles_labels()
        for artist, label in zip(artists, labels):
            handles.setdefault(label, artist)
    order = ['KC-1', 'SG-1b', 'SG-1', 'DHG', 'YH', 'HTTL', 'Smoothed regional trend', BAND_LABEL, 'Reconstructed range', 'Clipped to zero']
    labels = [label for label in order if label in handles]
    publication_legend(fig,[handles[label] for label in labels],labels)
    tab.axis('off')
    tab.set_title('Climate-cell-weighted RMSEP / predictive R²',pad=14)
    cores = sum(regions, [])
    rows = [[method] + [f'{stats[c]['cell_rmsep'][j]:.2f} / {stats[c]['cell_r2'][j]:.2f}' for c in cores] for j, method in enumerate(names)]
    columns=['Method']+[display_core(c) for c in cores]
    tbl = tab.table(cellText=rows, colLabels=columns, colWidths=[.22]+[.13]*6, cellLoc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
        cell.set_facecolor('#e1e9eb' if row == 0 else '#f2f5f5' if row % 2 else '#fafbfb')
        if row == 0:
            cell.get_text().set_weight('bold')
    details = dict(uncertainty=BAND_NOTE, methods=names, metrics=stats, predictions={c: v.tolist() for c, v in data.items()}, age={c: v['age'].tolist() for c, v in views.items()}, protocol=protocol + '; common test records across all ten methods per variable/core; identical observed targets and spatial fold assignments; fixed external plant distributions', display='West: KC-1, SG-1b, SG-1; East: DHG, YH, HTTL; 0-18.0 Ma; all DHG horizons within window; HTTL older than 5 Ma displayed; all fossil predictions retained; symbols show core-specific median age and median prediction per 0.25 Ma bin; independent panel y limits; 0.25 Ma within-core bins; within-core Gaussian sigma 0.6 Ma, cutoff 1.8 Ma; regional Gaussian sigma 0.4 Ma after linear interpolation across internal gaps; equal contributing cores; no extrapolation', intervals={c: [[a.tolist(), b.tolist()] for a, b in x] for c, x in ranges.items()})
    publication_finalize(fig)
    fig.savefig(out, dpi=300, metadata={'Description': json.dumps(add_definitions(details, 'MAT'))})
    plt.close(fig)
def redraw_cached():
    saved=load_cached(Path('D:/results/Supplementary Fig 4.png'))
    if saved['methods']!=names:
        raise ValueError('Cached method order differs from the code; use a matching source PNG.')
    views={core:{'age':np.asarray(age,float)} for core,age in saved['age'].items()}
    predictions={core:np.asarray(values,float) for core,values in saved['predictions'].items()}
    ranges={core:[(np.asarray(lo,float),np.asarray(hi,float)) for lo,hi in pairs]
            for core,pairs in saved['intervals'].items()}
    render(views,predictions,saved['metrics'],ranges)
if __name__=='__main__':
    import sys
    if '--refit' in sys.argv:
        main()
    else:
        redraw_cached()
