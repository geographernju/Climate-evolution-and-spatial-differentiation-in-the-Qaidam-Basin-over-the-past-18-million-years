import pandas as pd
import numpy as np
from pathlib import Path
input_path=Path('D:/data/5. KC-1.csv')
output_path=Path('D:/data/Processed data/borehole/KC-1.csv')
for encoding in ['utf-8-sig','utf-8','gb18030','cp1252','latin1']:
    try:
        df=pd.read_csv(input_path,encoding=encoding,low_memory=False)
        break
    except UnicodeDecodeError:
        continue
df.columns=[str(c).replace('\ufeff','').strip() for c in df.columns]
age_col='Age (ma)'
if age_col not in df.columns:
    raise KeyError(f'Cannot find {age_col}: {df.columns.tolist()}')
taxa_cols=[c for c in df.columns if c!=age_col]
counts=df[taxa_cols].apply(pd.to_numeric,errors='coerce')
if (counts<0).any().any():
    raise ValueError('Negative species counts detected. Please check the original data.')
total=counts.sum(axis=1,min_count=1)
relative=counts.div(total.replace(0,np.nan),axis=0)
result=pd.concat([df[[age_col]],relative],axis=1)
result.to_csv(output_path,index=False,encoding='utf-8-sig',float_format='%.12g')
print(f'Total samples: {len(df):,}')
print(f'Total species: {len(taxa_cols):,}')
print(f'Valid samples: {(total>0).sum():,}')
print(f'Zero-count samples: {(total==0).sum():,}')
print(f'Missing samples: {total.isna().sum():,}')
print(f'Relative abundance sum range: {relative.sum(axis=1)[total>0].min():.6f} - {relative.sum(axis=1)[total>0].max():.6f}')
print(f'Saved: {output_path}')
