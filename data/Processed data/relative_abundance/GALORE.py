import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
root=Path('D:/data/')
input_path=root/'2. GALORE_data.csv'
relative_path=root/'Processed data/relative_abundance/GALORE_relative_abundance.csv'
binary_path=root/'Processed data/relative_abundance/GALORE_presence_absence.csv'
def detect_encoding(path):
    for enc in ['utf-8-sig','utf-8','gb18030','cp1252','latin1']:
        try:
            with open(path,'r',encoding=enc) as f:
                while f.read(1024*1024):
                    pass
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode: {path}')
encoding=detect_encoding(input_path)
outputs={'Relative':relative_path,'Binary':binary_path}
for path in outputs.values():
    path.unlink(missing_ok=True)
written={key:False for key in outputs}
counts=Counter()
def save(df,key):
    if df.empty:
        return
    df.to_csv(outputs[key],mode='a' if written[key] else 'w',header=not written[key],index=False,encoding='utf-8' if written[key] else 'utf-8-sig',float_format='%.12g')
    written[key]=True
    counts[key]+=len(df)
for chunk in pd.read_csv(input_path,encoding=encoding,chunksize=200,low_memory=False):
    chunk.columns=[str(c).replace('\ufeff','').replace('ï»¿','').strip() for c in chunk.columns]
    metadata=chunk.iloc[:,:1].copy()
    taxa=chunk.columns[1:].tolist()
    pollen=chunk.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
    arr=pollen.to_numpy(dtype=float)
    finite=np.isfinite(arr)
    positive=finite&(arr>0)
    binary=(~finite|(arr==0)|(arr==1)).all(axis=1)&positive.any(axis=1)
    adjusted=np.where(finite,np.maximum(arr,0),np.nan)
    total=np.nansum(adjusted,axis=1)
    invalid=(total<=0)|np.isinf(arr).any(axis=1)
    valid=(~binary)&(~invalid)
    normalized=np.divide(adjusted,total[:,None],out=np.full_like(adjusted,np.nan),where=total[:,None]>0)
    relative=pd.concat([metadata,pd.DataFrame(normalized,columns=taxa,index=chunk.index)],axis=1)
    save(relative.loc[valid],'Relative')
    save(chunk.loc[binary],'Binary')
    counts['Invalid']+=int(invalid.sum())
print('\n========== GALORE Processing Summary ==========')
print(f'Relative abundance: {counts["Relative"]:,}')
print(f'Presence/absence: {counts["Binary"]:,}')
print(f'Invalid: {counts["Invalid"]:,}')
print(f'Total: {sum(counts.values()):,}')
print('\n========== Output Files ==========')
for path in outputs.values():
    print(path)
