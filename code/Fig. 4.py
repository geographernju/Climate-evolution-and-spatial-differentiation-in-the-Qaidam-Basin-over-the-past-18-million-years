import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')
from pathlib import Path
import argparse, hashlib, io, json, re, string, sys, zipfile, textwrap
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
sys.dont_write_bytecode = True
CATALOG, ERRORS, RAW = [], [], {}
COLORS = ['#24678c', '#b24d35', '#7257A6', '#298566', '#BD4D61']
def num(values):
    return pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
def find(root, *patterns):
    for pat in patterns:
        matches = sorted(set(root.glob(pat)) | set(root.glob('*/' + pat)))
        matches = [p for p in matches if p.is_file() and (not p.name.startswith('._')) and ('__MACOSX' not in str(p))]
        if matches:
            return matches[0]
    return None
def xls(path, sheet=0):
    return pd.read_excel(path, sheet_name=sheet, header=None, engine='openpyxl')
def tab(path):
    text = path.read_text(encoding='utf-8-sig')
    if '*/' in text:
        text = text.split('*/', 1)[1].lstrip()
    else:
        text = '\n'.join((l for l in text.splitlines() if l.strip() and (not l.startswith('#'))))
    return pd.read_csv(io.StringIO(text), sep='\t')
def series(key, path, sheet, ages, values, unit, mapping, lo=None, hi=None, note='', clip_to_window=True):
    frame = pd.DataFrame({'age_ma': num(ages), 'value': num(values)})
    if lo is not None:
        frame['lower'] = num(lo)
    if hi is not None:
        frame['upper'] = num(hi)
    before = len(frame)
    frame = frame[np.isfinite(frame.age_ma) & np.isfinite(frame.value)]
    if clip_to_window:
        frame = frame[(frame.age_ma >= 0) & (frame.age_ma <= 18)]
    frame = frame.sort_values('age_ma')
    if frame.empty:
        raise ValueError(key + ': no valid data in the 0-18 Ma interval')
    RAW[key] = frame
    CATALOG.append(dict(record=key, source=str(path), sheet=sheet, columns=mapping, unit=unit, n=len(frame), age_min_ma=float(frame.age_ma.min()), age_max_ma=float(frame.age_ma.max()), omitted_or_outside_0_18=before - len(frame), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), note=note))
    return frame
def attempt(name, function):
    try:
        return function()
    except Exception as error:
        ERRORS.append({'record': name, 'reason': str(error)})
        return None
def pair(root, key, patterns, sheet, skip, x, y, unit, mapping, scale=1, low=None, high=None, note=''):
    p = find(root, *patterns)
    if p is None:
        raise FileNotFoundError('Input file or data unavailable: ' + patterns[0])
    d = xls(p, sheet).iloc[skip:]
    return series(key, p, str(sheet), num(d.iloc[:, x]) * scale, d.iloc[:, y], unit, mapping, None if low is None else d.iloc[:, low], None if high is None else d.iloc[:, high], note)
def segments(d, gap=None):
    d = d.groupby('age_ma', as_index=False).median(numeric_only=True).sort_values('age_ma')
    if len(d) < 2:
        return [d]
    dif = np.diff(d.age_ma)
    positive = dif[dif > 0]
    if gap is None:
        gap = max(0.35, 5 * np.median(positive)) if len(positive) else 0.35
    cut = np.where(dif > gap)[0] + 1
    return [d.iloc[idx] for idx in np.split(np.arange(len(d)), cut) if len(idx)]
def curve(ax, d, color=COLORS[0], label=None, raw=False, band=False, ls='-', bin_width=None, gap=None, points_only=False):
    if d is None:
        return False
    if raw:
        ax.scatter(d.age_ma, d.value, s=4, color=color, alpha=0.2, linewidths=0, zorder=2, rasterized=True)
    if points_only:
        return True
    plot = d.copy()
    if bin_width:
        plot['bin'] = np.floor(plot.age_ma / bin_width).astype(int)
        plot = plot.groupby('bin')[['age_ma', 'value']].median().reset_index(drop=True)
    for j, s in enumerate(segments(plot, gap)):
        ax.plot(s.age_ma, s.value, color=color, lw=1.4, ls=ls, label=label if j == 0 else None, marker='o' if len(s) < 12 else None, ms=2.4, zorder=3)
        if band and {'lower', 'upper'} <= set(s.columns):
            valid = s[['lower', 'upper']].notna().all(axis=1) & (s.lower <= s.upper)
            ax.fill_between(s.age_ma, s.lower, s.upper, where=valid, color=color, alpha=0.13, linewidth=0, zorder=1)
    return True
def legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right', frameon=False, fontsize=11, ncol=min(3, len(labels)), handlelength=2)
def bin_core(age, value, grid):
    ok = np.isfinite(age) & np.isfinite(value)
    age = age[ok]
    value = value[ok]
    out = np.full(len(grid), np.nan)
    if not len(age):
        return out
    b = np.floor(age / 0.25)
    ub = np.unique(b)
    x = np.array([age[b == k].mean() for k in ub])
    y = np.array([value[b == k].mean() for k in ub])
    dd = grid[:, None] - x
    w = np.exp(-0.5 * (dd / 0.2) ** 2) * (abs(dd) <= 0.6)
    den = w.sum(1)
    ok = (den > 0) & (grid >= age.min()) & (grid <= age.max())
    out[ok] = w[ok] @ y / den[ok]
    return out
def regional(grid, trend, sigma):
    good = np.isfinite(trend)
    if not good.any():
        return (grid[:0], trend[:0])
    mask = (grid >= grid[good].min()) & (grid <= grid[good].max())
    x = grid[mask]
    return (x, gaussian_filter1d(np.interp(x, grid[good], trend[good]), sigma / (grid[1] - grid[0]), mode='nearest', truncate=3))
def original_band(parts, grid, x, center, sigma):
    rng = np.random.default_rng(20261115)
    draws = 2000
    total = np.zeros((len(grid), draws + 1))
    support = np.zeros(len(grid), int)
    for age, value in parts:
        good = np.isfinite(age) & np.isfinite(value)
        age = age[good]
        value = value[good]
        if not len(age):
            continue
        _, ix = np.unique(np.floor(age / 0.25), return_inverse=True)
        count = np.bincount(ix)
        xa = np.bincount(ix, weights=age) / count
        ya = np.bincount(ix, weights=value) / count
        dist = grid[:, None] - xa
        k = np.exp(-0.5 * (dist / 0.2) ** 2)
        k[abs(dist) > 0.6] = 0
        active = (grid >= age.min()) & (grid <= age.max()) & (k.sum(1) > 0)
        w = np.ones((len(count), draws + 1))
        w[:, 1:] = rng.exponential(size=(len(count), draws))
        kk = k[active]
        total[active] += kk * ya @ w / (kk @ w)
        support[active] += 1
    good = support > 0
    vals = total[good] / support[good, None]
    filled = np.column_stack([np.interp(x, grid[good], v) for v in vals.T])
    s = gaussian_filter1d(filled, sigma / (grid[1] - grid[0]), axis=0, mode='nearest', truncate=3)
    np.testing.assert_allclose(s[:, 0], center, rtol=1e-08, atol=1e-08)
    return np.quantile(s[:, 1:], [0.025, 0.975], axis=1)
def own_data(path):
    with Image.open(path) as im:
        saved = json.loads(im.info['Description'])
    for var in ['MAP','MAT']:
        reconstruction=saved[var]
        expected=['Modern Analogue','WA-PLS','RF','ExtraTrees','Boosting']
        if reconstruction.get('methods')!=expected:
            raise ValueError('Expected the ML_group_equal main reconstruction.')
        np.testing.assert_allclose(reconstruction['weights'],[1/3,1/3,1/9,1/9,1/9])
        for core,values in reconstruction['cores'].items():
            members=np.asarray([values['member_predictions'][m] for m in expected],float).T
            np.testing.assert_allclose(members@np.asarray(reconstruction['weights']),values['estimate'],equal_nan=True)
    grid = np.arange(0, 18.0125, 0.025)
    result = {}
    for field, var, unit in [('Annual precipitation (mm/year)', 'MAP', 'mm/yr'), ('Annual temperature (deg C)', 'MAT', '\xb0C')]:
        for names, region, sigma in [(['KC-1', 'SG-1b', 'SG-1'], 'West', 0.2), (['DHG', 'YH2', 'HTTL2'], 'East', 0.4)]:
            parts = []
            for name in names:
                v = saved.get(var,saved.get(field))['cores'][name]
                age = np.asarray(v['age'], float)
                y = np.asarray(v['estimate'], float)
                keep = (age >= 0) & (age <= 18) & (age > 5 if name == 'HTTL2' else True)
                parts.append((age[keep], y[keep]))
            curves = np.asarray([bin_core(age, y, grid) for age, y in parts])
            n = np.isfinite(curves).sum(0)
            trend = np.divide(np.nansum(curves, axis=0), n, out=np.full_like(grid, np.nan), where=n > 0)
            x, y = regional(grid, trend, sigma)
            lo, hi = original_band(parts, grid, x, y, sigma)
            key = f'Own_{region}_{var}'
            result[key] = series(key, path, 'PNG Description', x, y, unit, 'embedded core age+estimate; original mean-bin, core smoothing and regional algorithm', lo, hi, note='Exact trend and 2000-replicate bin-bootstrap interval recovered from PNG metadata; original internal interpolation retained.')
    return result
def style():
    plt.rcParams.update({'font.family':'Times New Roman', 'font.serif':['Times New Roman'], 'font.size':13,
        'mathtext.fontset':'custom', 'mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic', 'mathtext.bf':'Times New Roman:bold',
        'mathtext.bfit':'Times New Roman:bold:italic', 'mathtext.sf':'Times New Roman',
        'mathtext.tt':'Times New Roman', 'mathtext.cal':'Times New Roman',
        'mathtext.fallback':None,
        'axes.labelsize':14, 'xtick.labelsize':12, 'ytick.labelsize':12,
        'axes.unicode_minus':True, 'savefig.facecolor':'white'})
def panel_order(data,groups):
    priorities=[-1, -1, 1, 0, 2, 0, 2]
    group_names=['This study', 'This study', 'Atmospheric CO2', 'Global thermal and cryosphere context', 'North Pacific dust flux', 'Global thermal and cryosphere context', 'North Pacific dust flux']
    spans=[]
    for keys in groups:
        oldest=[];youngest=[]
        for key in keys:
            frame=data[key]
            if 'interval_ranges' in frame.attrs:
                intervals=frame.attrs['interval_ranges']
                oldest.extend(item['old_ma'] for item in intervals)
                youngest.extend(item['young_ma'] for item in intervals)
            else:
                oldest.append(float(frame.age_ma.max()));youngest.append(float(frame.age_ma.min()))
        spans.append(max(oldest)-min(youngest))
    order=[0,1]+sorted(range(2,len(groups)),key=lambda i:(priorities[i],-spans[i]))
    for i,keys in enumerate(groups):
        for key in keys:
            data[key].attrs['panel_span_myr']=spans[i]
            data[key].attrs['comparison_group']=group_names[i]
    return order
def remap_panels(used,order):
    return [(order.index(panel),key,label,kind,note) for panel,key,label,kind,note in used]
def panel_letter(order,old_index):
    return string.ascii_lowercase[order.index(old_index)]
def setup(title, labels, order=None):
    count = len(labels)
    order=list(range(count)) if order is None else order
    panel_height_in=0.94
    panel_gap_in=0.14
    top_margin_in=0.35
    bottom_margin_in=0.70
    height=top_margin_in+bottom_margin_in+count*panel_height_in+(count-1)*panel_gap_in
    fig, axes = plt.subplots(count, 1, figsize=(11.4,height), sharex=True,
                             gridspec_kw={'height_ratios':[1]*count})
    fig.subplots_adjust(left=.105, right=.895, top=1-top_margin_in/height,
                        bottom=bottom_margin_in/height, hspace=panel_gap_in/panel_height_in)
    logical_axes=np.empty(count,dtype=object)
    for i, ax in enumerate(axes):
        old_index=order[i];logical_axes[old_index]=ax
        name,unit=labels[old_index]
        ax.set_xlim(18, 0)
        ax.text(.012,.88,f'({string.ascii_lowercase[i]})',transform=ax.transAxes,
                fontsize=13,fontweight='bold',va='top',zorder=10,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1.1})
        ax.set_ylabel(unit, labelpad=11)
        ax.grid(axis='x', color='#DEE4EA', lw=.65)
        ax.spines[['top','left','right']].set_visible(False)
        side = 'left' if i % 2 == 0 else 'right'
        ax.yaxis.set_label_position(side)
        ax.yaxis.tick_left() if side == 'left' else ax.yaxis.tick_right()
        ax.spines[side].set_visible(True)
        ax.spines[[side,'bottom']].set_color('#ADB7C2')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='y', length=3)
        if i < count-1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)
    axes[-1].set_xlabel('Age (Ma)', fontsize=14)
    axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    return fig, logical_axes
def args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('D:/data/Processed data'))
    parser.add_argument('--own', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=Path('D:/results'))
    parser.add_argument('--dpi', type=int, default=300)
    result=parser.parse_args();result.out.mkdir(parents=True,exist_ok=True)
    if result.own is None:
        result.own=Path('D:/results/Fig. 2.png')
    return result
SOURCES = {
 'Tang2022': ('Tang et al. (2022), Geophysical Research Letters 49, e2022GL099311', 'DSDP 296 and ODP 1208; North Pacific', 'NOAA 36734 original eolian fluxes', 'https://doi.org/10.1029/2022GL099311', 'https://doi.org/10.25921/z6tc-4222'),
 'Herbert2016': ('Herbert et al. (2016), Nature Geoscience 9, 843-847',
         'ODP 982; North Atlantic', 'PANGAEA 885578: revised age model and published Muller et al. (1998) SST',
         'https://doi.org/10.1038/ngeo2813', 'https://doi.org/10.1594/PANGAEA.885578'),
 'Herbert2020': ('Herbert et al. (2020), Paleoceanography and Paleoclimatology 35, e2020PA003935',
         'Sdr. Vium borehole; Danish North Sea', 'PANGAEA 921243: sand-corrected BACON age and published Muller et al. (1998) SST',
         'https://doi.org/10.1029/2020PA003935', 'https://doi.org/10.1594/PANGAEA.921243'),
 'Own': ('This study', 'KC-1, SG-1b, SG-1 (West); DHG, YH2, HTTL2 (East)',
         'ML_group_equal.png: category-balanced main reconstruction; weights 1/3, 1/3, 1/9, 1/9, 1/9; embedded exact core predictions', 'Local final reconstruction', 'ML_group_equal.png (this study; archived exact core predictions)'),
 'Zhang': ('Zhang et al. (2024), Journal of Geophysical Research: Atmospheres',
           'KC-1, Qaidam; MJZ, Xining; YW, Longzhong', 'Supplementary Table S2; published LOWESS series',
           'https://doi.org/10.1029/2024JD041025','https://figshare.com/s/bc3ee24f430714dc5a6c'),
 'Wang2019': ('Wang et al. (2019), Nature Communications 10, 5272',
              'Lantian, Weihe Basin', 'Supplementary Table 5; coexistence approach plus ecosystem matching',
              'https://doi.org/10.1038/s41467-019-13338-4',
              'https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41467-019-13338-4/MediaObjects/41467_2019_13338_MOESM1_ESM.pdf'),
 'Jia': ('Jia et al. (2021), Palaeogeography, Palaeoclimatology, Palaeoecology 567, 110244',
         'KC-1, SG-1b, SG-1; western Qaidam Basin', 'jia2021.csv; ANNP_Mean and ANNT_Mean',
         'https://doi.org/10.1016/j.palaeo.2021.110244','Local author-supplied CSV'),
 'Bao': ('Bao et al. (2019), Journal of Asian Earth Sciences', 'Huaitoutala (HTTL), eastern Qaidam Basin',
         'Supplementary Table S1; Age and CIA columns; age >5 Ma',
         'https://doi.org/10.1016/j.jseaes.2018.10.018',
         'https://ars.els-cdn.com/content/image/1-s2.0-S1367912018304425-mmc1.docx'),
 'Liang': ('Liang et al. (2021), Palaeogeography, Palaeoclimatology, Palaeoecology, 110384',
           'KC-1; western Qaidam Basin', 'Supplementary biomarker workbook; beta-carotane/C36',
           'https://doi.org/10.1016/j.palaeo.2021.110384',
           'https://ars.els-cdn.com/content/image/1-s2.0-S0031018221001693-mmc1.xlsx'),
 'CO2': ('Cenozoic CO2 Proxy Integration Project Consortium (2023), Science',
         'Global compilation', 'Author Bayesian model output v1.2; 100-kyr CO2 posterior quantiles',
         'https://doi.org/10.1126/science.adi5177','https://zenodo.org/records/14537923'),
 'Westerhold': ('Westerhold et al. (2020), Science 369, 1383-1387',
                'Global benthic foraminiferal composite (CENOGRID)', 'PANGAEA 917717, Table S34; author long-term LOESS',
                'https://doi.org/10.1126/science.aba6853','https://doi.org/10.1594/PANGAEA.917717'),
 'Rea': ('Rea et al. (1998), Paleoceanography 13, 215-224', 'ODP Site 885A, North Pacific',
         'PANGAEA 859270; Age and dust mass accumulation rate',
         'https://doi.org/10.1029/98PA00123','https://doi.org/10.1594/PANGAEA.859270'),
}
def write_summary(out,figure,used):
    index={item['record']:item for item in CATALOG}
    indicators={
        'Jia_MAP':'Mean annual precipitation (MAP)',
        'Jia_MAT':'Mean annual temperature (MAT)',
        'Wang2019_MAP':'Mean annual precipitation range (MAP)',
        'Wang2019_MAT':'Mean annual temperature range (MAT)',
        'Chen2019_MAT':'brGDGT-derived annual air temperature (MATmr)',
        'Liang_beta':'Beta-carotane/n-C36',
        'Bao_CIA':'Chemical Index of Alteration (CIA)',
        'Westerhold_d18O':'Benthic foraminiferal delta18O',
        'CO2_fit':'Atmospheric CO2',
        'Herbert2016_SST':'Alkenone-derived sea-surface temperature (SST)',
        'Herbert2020_SST':'Alkenone-derived sea-surface temperature (SST)',
        'Tang2022_Flux296':'Eolian dust mass accumulation rate',
        'Tang2022_Flux1208':'Eolian dust mass accumulation rate',
        'Rea_DMAR':'Eolian dust mass accumulation rate',
    }
    titles={'10.1016/j.palaeo.2021.110244': 'Quantitative Cenozoic climatic reconstruction and its implications for aridification of the northeastern Tibetan Plateau', '10.1038/s41467-019-13338-4': 'Asian monsoon rainfall variation during the Pliocene forced by global temperature change', '10.1029/2024JD041025': 'Tectonically Controlled Establishment of Modern-Like Precipitation Patterns in East and Central Asia During the Early Late Miocene', '10.1029/2019GL082805': "A Late Miocene Terrestrial Temperature History for the Northeastern Tibetan Plateau's Period of Tectonic Expansion", '10.1016/j.palaeo.2021.110384': 'Evolution of the Miocene megalake in the western Qaidam Basin, northwestern China', '10.1016/j.jseaes.2018.10.018': 'Reduced chemical weathering intensity in the Qaidam Basin (NE Tibetan Plateau) during the Late Cenozoic', '10.1126/science.aba6853': "An astronomically dated record of Earth's climate and its predictability over the last 66 million years", '10.1038/ngeo2813': 'Late Miocene global cooling and the rise of modern ecosystems', '10.1029/2020PA003935': 'Bihemispheric Warming in the Miocene Climatic Optimum as Seen From the Danish North Sea', '10.1126/science.adi5177': 'Toward a Cenozoic history of atmospheric CO2', '10.1029/2022GL099311': 'Northward Shift of the Northern Hemisphere Westerlies in the Early to Late Miocene and Its Links to Tibetan Uplift', '10.1029/98PA00123': 'Late Cenozoic eolian deposition in the North Pacific: Asian drying, Tibetan uplift, and cooling of the northern hemisphere'}
    columns=['Subplots','Indicator','Site_or_region','Age_range_Ma','Units',
             'Reference','Publication_URL','Data_source']
    rows=[]
    for panel,key,label,kind,note in used:
        data=RAW[key];meta=index[key]
        source=next(value for prefix,value in SOURCES.items() if key.startswith(prefix))
        indicator=indicators.get(key,'Mean annual precipitation (MAP)')
        site=source[1]
        if key.startswith('Own_'):
            region=key.split('_')[1]
            indicator='Mean annual precipitation (MAP)' if key.endswith('_MAP') else 'Mean annual temperature (MAT)'
            site='West, Qaidam (KC-1, SG-1b, SG-1)' if region=='West' else 'East, Qaidam (DHG, YH2, HTTL2)'
        elif key.startswith('Zhang_'):
            site={'Zhang_Qaidam':'KC-1, Qaidam','Zhang_Xining':'MJZ, Xining','Zhang_Longzhong':'YW, Longzhong'}[key]
        elif key.startswith('Tang2022_'):
            site='DSDP 296, North Pacific' if key.endswith('296') else 'ODP 1208, North Pacific'
        age=f'{float(data.age_ma.max()):.4g} to {float(data.age_ma.min()):.4g}'
        if 'interval_ranges' in data.attrs:
            intervals=data.attrs['interval_ranges']
            age=f"{intervals[0]['old_ma']:g} to {intervals[-1]['young_ma']:g}"
        unit={'deg C':'deg C','°C':'deg C','mm/yr':'mm yr^-1','per mil VPDB':'per mil VPDB'}.get(meta['unit'],meta['unit'])
        full_reference='This study'
        if not key.startswith('Own_'):
            author,journal=source[0].split(', ',1)
            title=titles[source[3].removeprefix('https://doi.org/')]
            full_reference=f'{author}. {title}. {journal}.'
        rows.append(dict(Subplots=f'{figure}{string.ascii_lowercase[panel]}',Indicator=indicator,
                         Site_or_region=site,Age_range_Ma=age,Units=unit,
                         Reference=full_reference,
                         Publication_URL='' if key.startswith('Own_') else source[3],Data_source=source[4]))
    path=Path('D:/data/Processed data/Potential_driving_mechanisms/Indicator_summary.csv')
    if path.exists():
        previous=pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False)
        if list(previous.columns)!=columns:
            raise ValueError('Summary schema changed. Remove the old summary and run both plotting scripts to rebuild it.')
        for row in previous.to_dict('records'):
            panels=[p.strip() for p in row['Subplots'].split(';') if p.strip() and not p.strip().startswith(str(figure))]
            if panels:
                row['Subplots']='; '.join(panels);rows.append(row)
    combined={}
    for row in rows:
        identity=(row['Indicator'],row['Site_or_region'],row['Reference'])
        if identity in combined:
            existing=combined[identity]
            for field in columns[1:]:
                if str(existing[field])!=str(row[field]):
                    raise ValueError(f'Conflicting summary values for {identity}: {field}')
            panels=set(existing['Subplots'].split('; '))|set(row['Subplots'].split('; '))
            existing['Subplots']='; '.join(sorted(panels))
        else:
            combined[identity]=row.copy()
    table=pd.DataFrame(combined.values(),columns=columns)
    table.sort_values('Subplots',kind='stable').to_csv(path,index=False,encoding='utf-8-sig')
def load(root,own):
    data=own_data(own)
    mechanism_root=root/'Potential_driving_mechanisms'
    for site in ['296','1208']:
        path=mechanism_root/'Tang2022'/f'Tang2022_{site}eolianflux.txt';table=pd.read_csv(path,sep='\t',comment='#')
        key='Tang2022_Flux'+site
        data[key]=series(key,path,'NOAA original data',table.Age,table.Eolian_flux,'g cm^-2 kyr^-1','Age variable explicitly in Ma; Eolian_flux in g/cm2/kyr')
    mechanism_root=root/'Potential_driving_mechanisms'
    path=find(mechanism_root,'Herbert2016/*885578.tab');table=tab(path)
    data['Herbert2016_SST']=series('Herbert2016_SST',path,'PANGAEA 885578',num(table.iloc[:,5])/1000,table.iloc[:,9],
        'deg C','Published age [ka BP]/1000; SST Muller et al. (1998), source column 10',
        note='Repository revised age model (2017); published Muller SST; not local Qaidam MAT.')
    path=find(mechanism_root,'Herbert2020/*921243.tab');table=tab(path)
    data['Herbert2020_SST']=series('Herbert2020_SST',path,'PANGAEA 921243',num(table.iloc[:,3])/1000,table.iloc[:,6],
        'deg C','Sand-corrected BACON age [ka]/1000; SST Muller et al. (1998), source column 7',
        note='Sand-corrected published age model; 0-18 Ma only; replicate ages retained in raw observations.')
    path=find(root,'Westerhold2020/*917717.tab','Westerhold2020/datasets/TableS34.tab');table=tab(path)
    for name,rawcol,smoothcol in [('d18O',5,11)]:
        key='Westerhold_'+name
        data[key+'_raw']=series(key+'_raw',path,'Table S34',table.iloc[:,0],table.iloc[:,rawcol],'per mil VPDB',str(table.columns[rawcol]))
        data[key]=series(key,path,'Table S34',table.iloc[:,0],table.iloc[:,smoothcol],'per mil VPDB',str(table.columns[smoothcol]))
    path=find(root,'Rea1998/*859270.tab');table=tab(path)
    data['Rea_DMAR']=series('Rea_DMAR',path,'PANGAEA 859270',num(table['Age [ka BP]'])/1000,table['DMAR [g/cm**2/ka]'],'g cm^-2 kyr^-1','Age [ka BP]/1000; DMAR')
    path=find(root,'CenCO2PIP2023/*CO2_100kyr.csv','CenCO2PIP2023/*100kyrCO2.csv');table=pd.read_csv(path)
    data['CO2_fit']=series('CO2_fit',path,'100-kyr Bayesian model output',table.ages,np.exp(num(table['50%'])),'ppm',
                           'ages; exp(50%), exp(2.5%), exp(97.5%); original values are ln(CO2)',
                           np.exp(num(table['2.5%'])),np.exp(num(table['97.5%'])))
    return data
def main():
    a=args();style();data=load(a.root,a.own);used=[]
    labels=[('This study: final regional precipitation reconstructions','MAP (mm yr$^{-1}$)'),
            ('This study: final regional temperature reconstructions','MAT (°C)'),
            ('Atmospheric CO$_2$: published Bayesian fit - CenCO2PIP','CO$_2$ (ppm)'),
            ('Benthic oxygen isotopes - Westerhold et al. (2020; reversed axis)','$\\delta^{18}$O (\u2030 VPDB)'),
            ('Asian dust export: ODP 885A - Rea et al. (1998; reversed axis)','Dust MAR\n(g cm$^{-2}$ kyr$^{-1}$)'),
            ('North Atlantic and North Sea SST reconstructions','SST (°C)'),
            ('North Pacific paired dust fluxes','Dust MAR\n(g cm$^{-2}$ kyr$^{-1}$)')]
    groups=[['Own_West_MAP','Own_East_MAP'],['Own_West_MAT','Own_East_MAT'],['CO2_fit'],
            ['Westerhold_d18O'],['Rea_DMAR'],['Herbert2016_SST','Herbert2020_SST'],['Tang2022_Flux296','Tang2022_Flux1208']]
    order=panel_order(data,groups)
    fig,axes=setup('Figure 2. This study and potential climate-driver indicators',labels,order)
    def draw(panel,key,color,label,kind,note='',**kwargs):
        curve(axes[panel],data[key],color,label,**kwargs)
        used.append((panel,key,label,kind,note))
    for panel,var in [(0,'MAP'),(1,'MAT')]:
        for region,color in [('West',COLORS[0]),('East',COLORS[1])]:
            draw(panel,f'Own_{region}_{var}',color,region+', Qaidam','This study: final regional reconstruction',
                 'Same regional response series as Figure 1; shading: 95% bin-bootstrap interval.',band=True,gap=.1)
        legend(axes[panel])
    draw(2,'CO2_fit',COLORS[3],'Posterior median and 95% interval','Potential forcing: atmospheric CO2',
         'Author Bayesian fit on a 100-kyr grid; ln values exponentiated to ppm. Grid spacing is not independent temporal resolution.',band=True,gap=.15)
    for panel,key,label,color in [(3,'Westerhold_d18O','Benthic oxygen isotopes',COLORS[0])]:
        curve(axes[panel],data[key+'_raw'],'#AAB4BE',raw=True,points_only=True)
        draw(panel,key,color,label+' (published LOESS)','Isotope indicator',
             'Published LOESS. Benthic oxygen isotopes combine ice-volume and deep-water-temperature effects; not a direct surface-temperature reconstruction.')
    for panel,key,label,color,note in [
            (4,'Rea_DMAR','ODP 885A dust MAR',COLORS[4],'Sensitive to dust supply, aridity, wind and transport; not direct westerly strength. A dashed straight connector bridges the 8.24-9.08 Ma sampling gap; no observations are added.')]:
        draw(panel,key,color,label,'Potential-driver-related environmental indicator',note+' Reversed y-axis.',
             raw=True,bin_width=None)
    dust=data['Rea_DMAR'].groupby('age_ma',as_index=False).median(numeric_only=True).sort_values('age_ma')
    dust_steps=np.diff(dust.age_ma)
    dust_gap=max(.35,5*np.median(dust_steps[dust_steps>0]))
    for index in np.where(dust_steps>dust_gap)[0]:
        ends=dust.iloc[index:index+2]
        axes[4].plot(ends.age_ma,ends.value,color=COLORS[4],lw=1.25,ls='--',zorder=3)
    for panel in [3,4]:axes[panel].invert_yaxis()
    draw(5,'Herbert2016_SST',COLORS[0],'ODP 982','Large-scale ocean thermal background indicator',
         'Published Muller et al. (1998) SST and repository revised ages; raw points and 0.1-Myr bin medians; gaps >0.6 Myr unconnected. Not local Qaidam MAT.',
         raw=True,bin_width=.1,gap=.6)
    draw(5,'Herbert2020_SST',COLORS[1],'Sdr. Vium','Large-scale ocean thermal background indicator',
         'Published Muller et al. (1998) SST and sand-corrected BACON ages; no smoothing; line joins median replicates at identical ages; gaps >1.5 Myr unconnected. Not local Qaidam MAT.',
         raw=True,gap=1.5)
    axes[5].legend(loc='lower left',frameon=False,fontsize=11,ncol=2)
    for site,color in [('296',COLORS[1]),('1208',COLORS[2])]:
        draw(6,'Tang2022_Flux'+site,color,('DSDP ' if site=='296' else 'ODP ')+site,'Paired North Pacific dust-flux indicator',
             'NOAA original eolian fluxes and published ages; separate sites. Flux reflects provenance, source aridity, transport and circulation; not direct wind speed. Reversed y-axis.',raw=True,gap=.8)
    axes[6].invert_yaxis()
    axes[6].legend(loc='lower left',bbox_to_anchor=(.065,0),frameon=False,fontsize=11,ncol=2)
    fig.savefig(a.out/'Fig. 4.png',dpi=a.dpi);plt.close(fig)
    write_summary(a.out,2,remap_panels(used,order))
    print(f'Figure 2 complete: {len(used)} records.')
if __name__=='__main__':main()
