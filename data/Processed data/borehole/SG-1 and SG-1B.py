import pandas as pd
import numpy as np
from pathlib import Path
input_path=Path('D:/data/6. SG-1_and_SG-1b.csv')
output_dir=Path('D:/data/Processed data/borehole')
for encoding in ['utf-8-sig','utf-8','gb18030','cp1252','latin1']:
    try:
        df=pd.read_csv(input_path,encoding=encoding,low_memory=False)
        break
    except UnicodeDecodeError:
        continue
df.columns=[str(c).replace('\ufeff','').strip() for c in df.columns]
if 'Age (ka)' not in df.columns or 'Core' not in df.columns:
    raise KeyError(f'Required columns not found: {df.columns.tolist()}')
df['Age (ma)']=pd.to_numeric(df['Age (ka)'],errors='coerce')/1000
if df['Age (ma)'].isna().any():
    raise ValueError('Invalid or missing age values detected.')
df['Core']=df['Core'].astype(str).str.strip().str.lower().str.replace(r'\s+','',regex=True).str.replace('–','-',regex=False).str.replace('—','-',regex=False)
taxa_cols=sorted([c for c in df.columns if c not in ['Age (ka)','Age (ma)','Core']],key=str.casefold)
counts=df[taxa_cols].apply(pd.to_numeric,errors='coerce')
if (counts<0).any().any():
    raise ValueError('Negative species counts detected.')
df[taxa_cols]=counts
core_files={'sg-1':'SG-1.csv','sg-1b':'SG-1b.csv'}
unknown=set(df['Core'].unique())-set(core_files)
if unknown:
    raise ValueError(f'Unknown core names: {unknown}')
for core,filename in core_files.items():
    subset=df.loc[df['Core']==core].copy()
    if subset.empty:
        print(f'No records found for {core}')
        continue
    pollen=subset[taxa_cols]
    total=pollen.sum(axis=1,min_count=1)
    relative=pollen.div(total.replace(0,np.nan),axis=0)
    result=pd.concat([subset[['Age (ma)']],relative],axis=1)
    output_path=output_dir/filename
    result.to_csv(output_path,index=False,encoding='utf-8-sig',float_format='%.12g')
    valid=total>0
    sums=relative.sum(axis=1)
    print(f'\nCore: {core}')
    print(f'Total samples: {len(subset):,}')
    print(f'Total taxa: {len(taxa_cols):,}')
    print(f'Valid samples: {valid.sum():,}')
    print(f'Zero-count samples: {(total==0).sum():,}')
    print(f'Missing samples: {total.isna().sum():,}')
    print(f'Age range: {result["Age (ma)"].min():.6f} - {result["Age (ma)"].max():.6f} Ma')
    if valid.any():
        print(f'Relative abundance sum: {sums[valid].min():.6f} - {sums[valid].max():.6f}')
    print(f'Saved: {output_path}')
