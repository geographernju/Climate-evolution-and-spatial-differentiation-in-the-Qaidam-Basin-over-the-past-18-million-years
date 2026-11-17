import os
for _thread_var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_thread_var]='1'
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
REGIONS=[('West',['KC-1','SG-1b','SG-1']),('East',['DHG','YH2','HTTL2'])]
GRID=np.arange(0,18.0125,0.025)
METHODS=['Modern Analogue','WA','WA-PLS','RF','ExtraTrees','Boosting',
         'CA','MCR-w','P-CRACLE','CREST proxy','Ensemble']
METHOD_COLUMNS=['Modern_Analogue','WA','WA_PLS','RF','ExtraTrees','Boosting',
                'CA','MCR_w','P_CRACLE','CREST_proxy','Ensemble']
GROUPS=[('West','MAP'),('East','MAP'),('West','MAT'),('East','MAT')]
def cached(path):
    with Image.open(path) as im:
        return json.loads(im.info['Description'])
def shown(core,age):
    return np.isfinite(age)&(age>=0)&(age<=18)&((age>5) if core=='HTTL2' else True)
def core_smooth(age,value,sigma):
    good=np.isfinite(age)&np.isfinite(value)
    age,value=age[good],value[good]
    result=np.full(len(GRID),np.nan)
    if not len(age):return result
    bins=np.floor(age/.25).astype(int)
    unique=np.unique(bins)
    x=np.array([age[bins==b].mean() for b in unique])
    y=np.array([value[bins==b].mean() for b in unique])
    distance=GRID[:,None]-x
    kernel=np.exp(-.5*(distance/sigma)**2)*(abs(distance)<=round(3*sigma,8))
    total=kernel.sum(1)
    good=(total>0)&(GRID>=age.min())&(GRID<=age.max())
    result[good]=kernel[good]@y/total[good]
    return result
def regional_curve(parts,core_sigma,region_sigma):
    smooth=np.array([core_smooth(x,y,core_sigma) for x,y in parts])
    count=np.isfinite(smooth).sum(0)
    mean=np.divide(np.nansum(smooth,axis=0),count,
                   out=np.full(len(GRID),np.nan),where=count>0)
    support=np.isfinite(mean)
    if not support.any():return np.array([]),np.array([])
    keep=(GRID>=GRID[support].min())&(GRID<=GRID[support].max())
    age=GRID[keep]
    values=np.interp(age,GRID[support],mean[support])
    return age,gaussian_filter1d(values,region_sigma/(GRID[1]-GRID[0]),
                                  mode='nearest',truncate=3)
def export(root,output):
    curves={}
    def add(var,method,series,ensemble=False):
        for region,cores in REGIONS:
            parts=[]
            for core in cores:
                age,value=series[core]
                keep=shown(core,age)
                ix=np.flatnonzero(keep)[np.argsort(age[keep],kind='stable')]
                parts.append((age[ix],value[ix]))
            core_sigma=.2 if ensemble else .6
            region_sigma=(.2 if region=='West' else .4) if ensemble else .4
            age,value=regional_curve(parts,core_sigma,region_sigma)
            column=f'{region}_{var}_{METHOD_COLUMNS[METHODS.index(method)]}'
            if column in curves:raise ValueError(f'Duplicate reconstruction: {column}')
            curves[column]={round(float(x),10):float(y) for x,y in zip(age,value)}
    figure_files = ['Supplementary Fig 1.png','Supplementary Fig 2.png','Supplementary Fig 3.png',
                    'Supplementary Fig 4.png','Supplementary Fig 5.png','Supplementary Fig 6.png']
    for filename in figure_files:
        d=cached(root/filename)
        for j,method in enumerate(d['methods']):
            series={core:(np.asarray(d['age'][core],float),np.asarray(values,float)[:,j])
                    for core,values in d['predictions'].items()}
            add(d['climate_variable'],method,series)
    final=cached(root/'Fig. 2.png')
    for var in ['MAP','MAT']:
        assert final[var]['methods']==['Modern Analogue','WA-PLS','RF','ExtraTrees','Boosting']
        np.testing.assert_allclose(final[var]['weights'],[1/3,1/3,1/9,1/9,1/9])
        for core,values in final[var]['cores'].items():
            members=np.asarray([values['member_predictions'][name] for name in final[var]['methods']],float).T
            expected=members@np.asarray(final[var]['weights'],float)
            np.testing.assert_allclose(expected,values['estimate'],rtol=1e-12,atol=1e-10,equal_nan=True)
        series={core:(np.asarray(v['age'],float),np.asarray(v['estimate'],float))
                for core,v in final[var]['cores'].items()}
        add(var,'Ensemble',series,ensemble=True)
    columns=[f'{region}_{var}_{method}' for region,var in GROUPS for method in METHOD_COLUMNS]
    if set(curves)!=set(columns):raise ValueError('Incomplete set of reconstruction curves')
    ages=sorted({x for values in curves.values() for x in values},reverse=True)
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Age_Ma']+columns)
        writer.writeheader()
        for age in ages:
            row={'Age_Ma':age}
            row.update({column:curves[column].get(age,'') for column in columns})
            writer.writerow(row)
    print(f'Saved {len(ages)} ages, four climate groups, 11 methods per group: {output}')
def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,default=Path('D:/results'))
    parser.add_argument('--output',type=Path,default=Path('D:/data/10.Climate_reconstruction_summary.csv'))
    a=parser.parse_args();export(a.root,a.output)
if __name__=='__main__':main()
