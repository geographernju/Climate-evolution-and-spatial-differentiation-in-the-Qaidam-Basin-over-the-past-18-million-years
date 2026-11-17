import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import unicodedata
sift_path=Path(r'D:\data\Processed data\relative_abundance\GALORE sift\GALORE_relative_abundance_sift.csv')
loc_path=Path(r'D:\data\1. GALORE_locations.csv')
out_path=Path(r'D:\data\Processed data\relative_abundance\GALORE sift\distribution.png')
def read_table(path):
    if path.suffix.lower() in ('.xlsx','.xls'):
        df=pd.read_excel(path)
    else:
        for enc in ('utf-8-sig','utf-8','gb18030','cp1252','latin1'):
            try:
                df=pd.read_csv(path,encoding=enc,low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'Cannot read {path}')
    df.columns=[unicodedata.normalize('NFKC',str(c)).replace('\ufeff','').replace('ï»¿','').strip() for c in df.columns]
    return df
sift=read_table(sift_path)
loc=read_table(loc_path)
if 'Name' not in sift.columns or not {'Name','Longitude','Latitude'}.issubset(loc.columns):
    raise KeyError('Required columns not found')
sift['Name']=sift['Name'].astype('string').str.strip()
loc['Name']=loc['Name'].astype('string').str.strip()
taxa_cols=[c for c in sift.columns if c!='Name']
pollen=sift[taxa_cols].apply(pd.to_numeric,errors='coerce')
sift['Valid_Taxa']=(pollen>0).sum(axis=1)
loc_sub=loc[['Name','Longitude','Latitude']].copy()
loc_sub['Longitude']=pd.to_numeric(loc_sub['Longitude'],errors='coerce')
loc_sub['Latitude']=pd.to_numeric(loc_sub['Latitude'],errors='coerce')
loc_sub=loc_sub.dropna(subset=['Name','Longitude','Latitude'])
loc_sub=loc_sub.loc[loc_sub['Longitude'].between(-180,360)&loc_sub['Latitude'].between(-90,90)].copy()
loc_sub['Longitude']=(loc_sub['Longitude']+180)%360-180
loc_sub=loc_sub.drop_duplicates(subset=['Name','Longitude','Latitude'])
ambiguous=(loc_sub.groupby('Name').size()>1).sum()
if ambiguous:
    print(f'Warning: {ambiguous} names have multiple coordinate locations; first location will be used')
loc_sub=loc_sub.drop_duplicates(subset='Name',keep='first')
merged=sift[['Name','Valid_Taxa']].merge(loc_sub,on='Name',how='left',validate='many_to_one')
unmatched=merged['Longitude'].isna().sum()
merged=merged.dropna(subset=['Longitude','Latitude']).copy()
groups=[('10–14',10,15,'#3979b7',13),('15–19',15,20,'#ee9d40',17),('≥20',20,np.inf,'#cf383b',22)]
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=20
plt.rcParams['axes.unicode_minus']=False
fig=plt.figure(figsize=(16,8.5))
ax=fig.add_axes([0.015,0.035,0.97,0.93],projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.OCEAN.with_scale('110m'),facecolor='#eef5fa',edgecolor='none',zorder=0)
ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#e6e6e6',edgecolor='none',zorder=1)
ax.coastlines(resolution='110m',linewidth=0.55,color='#555555',zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale('110m'),linewidth=0.3,edgecolor='#999999',zorder=2)
ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,linewidth=0.4,color='gray',alpha=0.3,linestyle='--',zorder=1)
print('========== Sample statistics ==========')
print(f'Total samples: {len(sift):,}')
print(f'Unmatched coordinates: {unmatched:,}')
print(f'Valid coordinates: {len(merged):,}')
print(f'Excluded (<10 taxa): {(merged["Valid_Taxa"]<10).sum():,}')
for label,lower,upper,color,size in groups:
    subset=merged.loc[(merged['Valid_Taxa']>=lower)&(merged['Valid_Taxa']<upper)]
    ax.scatter(subset['Longitude'],subset['Latitude'],s=size,c=color,alpha=0.78,edgecolors='none',transform=ccrs.PlateCarree(),rasterized=True,zorder=3+lower,label=f'{label} taxa (n={len(subset):,})')
    print(f'{label} taxa: {len(subset):,}')
ax.legend(loc='lower left',bbox_to_anchor=(0.04,0.03),fontsize=17,frameon=True,facecolor='white',edgecolor='#bbbbbb',framealpha=0.95,markerscale=1.7,labelspacing=0.55,handletextpad=0.4)
ax.spines['geo'].set_linewidth(0.8)
fig.savefig(out_path,dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.close(fig)
print(f'Saved: {out_path}')
