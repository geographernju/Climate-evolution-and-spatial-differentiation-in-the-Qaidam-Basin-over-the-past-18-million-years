import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from copy import copy
ROOT=Path('D:/data')
SOURCE=ROOT/'7. DHG.csv'
NLR_PATH=ROOT/'3. NLR.xlsx'
GALORE_PATH=ROOT/'Processed data/relative_abundance/GALORE sift/GALORE_relative_abundance_sift.csv'
OUT=ROOT/'Processed data/borehole/DHG.csv'
AGE='Age (Ma)'
STD='Standardized pollen taxon'
NLR='Nearest living relative (NLR)'
RANK='NLR rank'
GALORE='GALORE comparable taxon'
CONF='Mapping confidence'
NOTES='Taxonomic notes'
REF='Reference URL'
SITE='DHG'
MAPPINGS=[
    ('Compositae','Asteraceae','Family',['Asteraceae'],'Direct taxon-level match','Historical family synonym.'),
    ('Artemisiaepollenites','Artemisia','Genus',['Artemisia'],'Qualified','Provisional fossil pollen affinity.'),
    ('Ephedrapites','Ephedra','Genus',['Ephedraceae','Ephedra'],'Qualified','Fossil pollen affinity; family-level comparison where required.'),
    ('Nitrariadites','Nitraria','Genus',['Nitraria'],'Qualified','Fossil Nitraria-type pollen.'),
    ('Qinghaipollis','Apiaceae','Family',['Apiaceae'],'Caution','Tentative family-level botanical affinity.'),
    ('Gramineae','Poaceae','Family',['Poaceae','Gramineae'],'Direct taxon-level match','Historical family synonym.'),
    ('Pinuspollenites','Pinus','Genus',['Pinus'],'Qualified','Fossil pollen affinity.'),
    ('Piceapollenites','Picea','Genus',['Picea'],'Qualified','Fossil pollen affinity.'),
    ('Abiespollenites','Abies','Genus',['Abies'],'Qualified','Fossil pollen affinity.'),
    ('Taxodiaceae','Taxodiaceae sensu lato','Family',['Cupressaceae','Taxodiaceae'],'Qualified','Historical family included in modern Cupressaceae.'),
    ('Podocarpidites','Podocarpus','Genus',['Podocarpus'],'Qualified','Provisional fossil pollen affinity.'),
    ('Tsugaepollenites','Tsuga','Genus',['Tsuga'],'Qualified','Fossil pollen affinity.'),
    ('Meliaceopollenites','Meliaceae','Family',['Meliaceae'],'Caution','Provisional family-level affinity.'),
    ('Ilexpollenites','Ilex','Genus',['Ilex'],'Qualified','Fossil pollen affinity.'),
    ('Salixipollenites','Salix','Genus',['Salix'],'Qualified','Fossil pollen affinity.'),
    ('Betulaceoipollenites','Betulaceae','Family',['Betulaceae','Betula'],'Caution','Prefer family-level correspondence; Betula is a tentative alternative.'),
    ('Aceripollenites','Acer','Genus',['Acer'],'Qualified','Fossil pollen affinity.'),
    ('Juglanspollenites','Juglans','Genus',['Juglans'],'Qualified','Provisional fossil pollen affinity.'),
    ('Ulmipollenites','Ulmus','Genus',['Ulmus'],'Qualified','Fossil pollen affinity.'),
    ('Quercopollenites','Quercus','Genus',['Quercus'],'Qualified','Fossil pollen affinity.')
]
FIXED={'compositae':'Asteraceae','qinghaipollis':'Apiaceae'}
FAMILY_NAMES={'Asteraceae','Apiaceae','Poaceae','Gramineae','Ephedraceae','Cupressaceae','Taxodiaceae','Betulaceae','Meliaceae'}
def clean(value):
    return '' if value is None else str(value).strip()
def key(value):
    return clean(value).casefold()
def read_csv_auto(path,**kwargs):
    for encoding in ['utf-8-sig','gb18030','latin1']:
        try:
            return pd.read_csv(path,encoding=encoding,**kwargs)
        except UnicodeError:
            continue
    raise ValueError(f'Cannot read: {path}')
for path in [SOURCE,NLR_PATH,GALORE_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)
df=read_csv_auto(SOURCE,float_precision='round_trip')
df.columns=df.columns.astype(str).str.strip()
df=df.loc[:,~df.columns.str.startswith('Unnamed:')]
if AGE not in df.columns:
    raise ValueError('Age column not found')
taxa=[c for c in df.columns if c!=AGE]
if len(df)!=76 or len(taxa)!=22:
    raise ValueError(f'Unexpected dimensions: {len(df)} layers, {len(taxa)} taxa')
if len(set(map(key,taxa)))!=len(taxa):
    raise ValueError('Duplicate input taxon names')
df[AGE]=pd.to_numeric(df[AGE],errors='raise')
df[taxa]=df[taxa].apply(pd.to_numeric,errors='raise')
df=df.sort_values(AGE).reset_index(drop=True)
if not np.isfinite(df[AGE]).all():
    raise ValueError('Invalid age values')
if df[AGE].duplicated().any():
    raise ValueError('Duplicate ages detected')
if np.isinf(df[taxa].to_numpy(float)).any() or (df[taxa]<0).any().any():
    raise ValueError('Invalid pollen abundances')
original_count=int(df[taxa].notna().sum().sum())
original_sums=df[taxa].sum(axis=1).to_numpy()
if original_count!=783:
    raise ValueError(f'Expected 783 pollen observations, found {original_count}')
modern_columns=read_csv_auto(GALORE_PATH,nrows=0).columns.astype(str).str.strip().tolist()
modern_map={key(c):c for c in modern_columns}
wb=load_workbook(NLR_PATH)
ws=wb['NLR Mapping']
headers={clean(c.value):c.column for c in ws[1] if c.value is not None}
required=[STD,NLR,RANK,GALORE,CONF,'KC-1','SG-1','SG-1b',NOTES,REF]
for name in required:
    if name not in headers:
        raise ValueError(f'Missing NLR column: {name}')
if SITE not in headers:
    col=ws.max_column+1
    ws.cell(1,col,SITE)
    ws.cell(1,col)._style=copy(ws.cell(1,headers['SG-1b'])._style)
    ws.column_dimensions[get_column_letter(col)].width=12
    headers[SITE]=col
for row in range(2,ws.max_row+1):
    ws.cell(row,headers[SITE]).value=None
def get_value(row,column):
    return clean(ws.cell(row,headers[column]).value)
existing={}
for row in range(2,ws.max_row+1):
    name=get_value(row,STD)
    if name:
        existing.setdefault(key(name),[]).append(row)
def mark_site(name):
    for row in existing.get(key(name),[]):
        ws.cell(row,headers[SITE]).value='Yes'
def existing_target(name):
    targets={get_value(row,GALORE) for row in existing.get(key(name),[]) if get_value(row,GALORE)}
    if len({key(v) for v in targets})>1:
        raise ValueError(f'Conflicting NLR mappings: {name}')
    return next(iter(targets),'')
def update_record(row,record):
    for column,value in record.items():
        ws.cell(row,headers[column]).value=value
def add_record(name,nlr,rank,target,confidence,notes,reference=None):
    if key(name) in existing:
        mark_site(name)
        return existing[key(name)][0]
    row=ws.max_row+1
    record={STD:name,NLR:nlr,RANK:rank,GALORE:target or None,CONF:confidence,NOTES:notes,REF:reference,SITE:'Yes'}
    for column in headers:
        cell=ws.cell(row,headers[column])
        cell._style=copy(ws.cell(2,headers[column])._style)
    update_record(row,record)
    existing[key(name)]=[row]
    return row
def ensure_modern_record(name,rank,force=False):
    actual=modern_map.get(key(name),'')
    if key(name) not in existing:
        add_record(name,name,rank,actual,'Direct taxon-level match','Modern comparison taxon.')
    elif force:
        for row in existing[key(name)]:
            update_record(row,{NLR:name,RANK:rank,GALORE:actual or None,CONF:'Direct taxon-level match'})
    mark_site(name)
    return existing_target(name)
mapping_dict={key(item[0]):item for item in MAPPINGS}
rename_map={}
added_fossil=0
added_modern=0
for taxon in taxa:
    k=key(taxon)
    if k not in mapping_dict:
        if k in existing:
            mark_site(taxon)
            rename_map[taxon]=get_value(existing[k][0],STD)
        else:
            rename_map[taxon]=taxon
        continue
    fossil,nlr,rank,candidates,confidence,notes=mapping_dict[k]
    target=next((modern_map[key(c)] for c in candidates if key(c) in modern_map),'')
    if k not in existing:
        add_record(fossil,nlr,rank,target,confidence,notes)
        added_fossil+=1
    elif k in FIXED:
        for row in existing[k]:
            update_record(row,{NLR:nlr,RANK:rank,GALORE:target or None,CONF:confidence,NOTES:notes,REF:None,SITE:'Yes'})
    else:
        old_target=existing_target(fossil)
        if old_target:
            target=modern_map.get(key(old_target),'')
        elif target:
            for row in existing[k]:
                ws.cell(row,headers[GALORE]).value=target
        mark_site(fossil)
    if k in FIXED:
        final_name=FIXED[k]
        if key(final_name) not in existing:
            added_modern+=1
        ensure_modern_record(final_name,'Family',force=True)
        rename_map[taxon]=final_name
        continue
    if not target:
        rename_map[taxon]=fossil
        continue
    target_rank='Family' if target in FAMILY_NAMES else 'Genus'
    if key(target) not in existing:
        added_modern+=1
    alias_target=ensure_modern_record(target,target_rank)
    if alias_target==target:
        rename_map[taxon]=get_value(existing[key(target)][0],STD)
    else:
        rename_map[taxon]=fossil
for name in ['Asteraceae','Apiaceae']:
    mark_site(name)
new_names=[rename_map.get(c,c) for c in taxa]
if len(set(map(key,new_names)))!=len(new_names):
    duplicates=[name for name in new_names if new_names.count(name)>1]
    raise ValueError(f'Duplicate output taxa: {duplicates}')
result=df.rename(columns=rename_map)
taxa_sorted=sorted([c for c in result.columns if c!=AGE],key=str.casefold)
result=result[[AGE]+taxa_sorted]
final_count=int(result[taxa_sorted].notna().sum().sum())
final_sums=result[taxa_sorted].sum(axis=1).to_numpy()
if len(result)!=76 or len(taxa_sorted)!=22:
    raise ValueError('Output dimensions changed')
if final_count!=original_count:
    raise ValueError('Pollen observation count changed')
if not np.allclose(original_sums,final_sums,rtol=0,atol=1e-8):
    raise ValueError('Layer abundance sums changed')
for taxon in taxa:
    new_name=rename_map.get(taxon,taxon)
    before=df[taxon].to_numpy(float)
    after=result[new_name].to_numpy(float)
    if not np.array_equal(before,after,equal_nan=True):
        raise ValueError(f'Pollen values changed: {taxon}')
matched=[]
unmatched=[]
for taxon in taxa_sorted:
    target=existing_target(taxon)
    if not target:
        target=taxon
    if target in modern_columns:
        matched.append(taxon)
    else:
        unmatched.append(taxon)
coverage=result[matched].sum(axis=1).div(result[taxa_sorted].sum(axis=1).replace(0,np.nan))
if ws.auto_filter.ref:
    ws.auto_filter.ref=ws.dimensions
OUT.parent.mkdir(parents=True,exist_ok=True)
wb.save(NLR_PATH)
result.to_csv(OUT,index=False,encoding='utf-8-sig')
print('========== Taxon renaming ==========')
for old,new in rename_map.items():
    if old!=new:
        print(f'{old:25s} -> {new}')
print('========== Final statistics ==========')
print(f'Total layers: {len(result)}')
print(f'Total taxa: {len(taxa_sorted)}')
print(f'Original pollen observations: {original_count}')
print(f'Final pollen observations: {final_count}')
print(f'Added fossil NLR records: {added_fossil}')
print(f'Added modern NLR records: {added_modern}')
print(f'Matched GALORE taxa: {len(matched)}')
print(f'Unmatched GALORE taxa: {len(unmatched)}')
print(f'Mean abundance coverage: {coverage.mean()*100:.3f}%')
print(f'Minimum abundance coverage: {coverage.min()*100:.3f}%')
print(f'Minimum layer abundance: {final_sums.min():.3f}%')
print(f'Mean layer abundance: {final_sums.mean():.3f}%')
print(f'Maximum layer abundance: {final_sums.max():.3f}%')
print('========== Unmatched GALORE taxa ==========')
for taxon in unmatched:
    print(taxon)
print(f'Saved: {NLR_PATH}')
print(f'Saved: {OUT}')
