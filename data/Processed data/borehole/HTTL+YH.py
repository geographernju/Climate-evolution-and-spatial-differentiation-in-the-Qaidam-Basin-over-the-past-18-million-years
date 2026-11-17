from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('D:/data')
OUTPUT=ROOT/'Processed data/borehole'
SOURCES={'YH':ROOT/'8. YH.csv','HTTL':ROOT/'9. HTTL.csv'}
NAME_MAPPING={'Picea+Abies':'Picea','Ephedra':'Ephedraceae','Betulaceae':'Betula','Amaranthaceae':'Chenopodiaceae','Asteraceae':'Aster'}
EXCLUDED={'Y','Age (Ma)','Age','Depth','Thickness','Core','Sample','Sample ID','Total counts','Sum spores'}
def process(path):
    data=pd.read_csv(path,encoding='utf-8-sig')
    data.columns=data.columns.astype(str).str.strip()
    if 'Age (Ma)' not in data:
        raise ValueError(f'Missing Age (Ma) column: {path}')
    ages=pd.to_numeric(data['Age (Ma)'],errors='raise')
    if not np.isfinite(ages).all():
        raise ValueError(f'Invalid ages: {path}')
    columns=[c for c in data.columns if c not in EXCLUDED and not c.startswith('Unnamed:')]
    pollen=data[columns].apply(pd.to_numeric,errors='raise')
    if np.isinf(pollen.to_numpy()).any() or (pollen<0).any().any():
        raise ValueError(f'Invalid pollen values: {path}')
    pollen=pollen.rename(columns=NAME_MAPPING).T.groupby(level=0,sort=True).sum(min_count=1).T
    totals=pollen.sum(axis=1,min_count=1)
    if totals.isna().any() or (totals<=0).any():
        raise ValueError(f'Samples without positive pollen totals: {path}')
    relative=pollen.div(totals,axis=0)
    np.testing.assert_allclose(relative.sum(axis=1),1,rtol=1e-12,atol=1e-12)
    return pd.concat([ages,relative],axis=1)
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for site,path in SOURCES.items():
        result=process(path)
        destination=OUTPUT/f'{site}.csv'
        result.to_csv(destination,index=False,encoding='utf-8-sig',float_format='%.6f')
        print(f'Saved {len(result)} samples and {len(result.columns)-1} taxa: {destination}')
if __name__=='__main__':
    main()
