import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
input_path=Path(r'D:\data\Processed data\relative_abundance\GALORE_relative_abundance.csv')
nlr_path=Path(r'D:\data\3. NLR.xlsx')
output_dir=Path(r'D:\data\Processed data\relative_abundance\GALORE sift')
output_dir.mkdir(parents=True,exist_ok=True)
output_path=output_dir/'GALORE_relative_abundance_sift.csv'
min_taxa=10
chunk_size=500
def clean_name(s):
    return unicodedata.normalize('NFKC',str(s)).replace('\ufeff','').replace('ï»¿','').strip()
def match_key(s):
    return re.sub(r'\s+',' ',clean_name(s)).casefold()
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
nlr=pd.read_excel(nlr_path,engine='openpyxl')
nlr.columns=[clean_name(c) for c in nlr.columns]
target_col='GALORE comparable taxon'
if target_col not in nlr.columns:
    raise KeyError(f'Column not found: {target_col}')
skip={'','nan','none','unresolved','unknown','not available','not found','no match','n/a','na','-'}
targets={}
for value in nlr[target_col].dropna():
    for item in re.split(r'[;\n；]+',str(value)):
        name=clean_name(item)
        key=match_key(name)
        if key not in skip and key not in targets:
            targets[key]=name
encoding=detect_encoding(input_path)
source_columns=pd.read_csv(input_path,encoding=encoding,nrows=0).columns.tolist()
sample_col=source_columns[0]
source_map={}
for col in source_columns[1:]:
    key=match_key(col)
    if key in source_map:
        raise ValueError(f'Duplicate taxon name: {col} / {source_map[key]}')
    source_map[key]=col
matched=[]
missing=[]
for key,name in targets.items():
    if key in source_map:
        matched.append((name,source_map[key]))
    else:
        missing.append(name)
matched.sort(key=lambda x:x[0].casefold())
if not matched:
    raise ValueError('No target taxa matched the GALORE columns')
selected_names=[x[0] for x in matched]
selected_cols=[x[1] for x in matched]
print('========== Taxon matching ==========')
print(f'NLR target taxa: {len(targets)}')
print(f'Matched taxa: {len(matched)}')
print(f'Unmatched taxa: {len(missing)}')
if missing:
    print('Unmatched taxon names:')
    for name in missing:
        print(f'  {name}')
print('\n========== Data processing ==========')
total_rows=0
retained_rows=0
written=False
for chunk in pd.read_csv(input_path,encoding=encoding,usecols=[sample_col]+selected_cols,chunksize=chunk_size,low_memory=False):
    total_rows+=len(chunk)
    pollen=chunk[selected_cols].apply(pd.to_numeric,errors='coerce')
    if (pollen<0).any().any():
        raise ValueError('Negative abundance detected in the input data')
    pollen.columns=selected_names
    positive_count=(pollen>0).sum(axis=1)
    keep=positive_count>=min_taxa
    if not keep.any():
        continue
    filtered=pollen.loc[keep].copy()
    totals=filtered.sum(axis=1,min_count=1)
    filtered=filtered.div(totals,axis=0)
    result=pd.concat([chunk.loc[keep,[sample_col]],filtered],axis=1)
    result.columns=['Name']+selected_names
    result.to_csv(output_path,mode='a' if written else 'w',header=not written,index=False,encoding='utf-8' if written else 'utf-8-sig',float_format='%.12g')
    written=True
    retained_rows+=len(result)
    print(f'Processed: {total_rows:,} | Retained: {retained_rows:,}',end='\r')
if not written:
    pd.DataFrame(columns=['Name']+selected_names).to_csv(output_path,index=False,encoding='utf-8-sig')
print('\n========== Final summary ==========')
print(f'Original samples: {total_rows:,}')
print(f'Retained samples: {retained_rows:,}')
print(f'Removed samples: {total_rows-retained_rows:,}')
print(f'Retention rate: {retained_rows/total_rows*100:.2f}%' if total_rows else 'Retention rate: N/A')
print(f'Target taxa: {len(targets)}')
print(f'Matched taxa: {len(matched)}')
print(f'Minimum nonzero taxa: {min_taxa}')
print(f'Saved: {output_path}')
