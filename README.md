# Climate Reconstruction Data and Figures

This guide covers `D:\code`, `D:\data`, and `D:\results`.

## Directories and files

| Directory | Purpose |
|---|---|
| `D:\code` | Python scripts for main and supplementary figures, plus the Fig. 1 PSD file |
| `D:\data` | Raw data, taxonomic mappings, processed data, preprocessing scripts, and the reconstruction summary |
| `D:\results` | PNG outputs; some PNG files also contain cached reconstruction data |

### Raw data and supporting information

| File in `D:\data` | Purpose |
|---|---|
| `1. GALORE_locations.csv` | GALORE sample coordinates |
| `1. GALORE_locations.png` | Corresponding sample location map |
| `2. GALORE_data.csv` | Raw GALORE pollen data |
| `3. NLR.xlsx` | Pollen taxonomy, nearest living relatives, and GALORE mappings |
| `4. Borehole_information.csv` | Borehole information and locations |
| `5. KC-1.csv` | Raw KC-1 data |
| `6. SG-1_and_SG-1b.csv` | Raw SG-1 and SG-1b data |
| `7. DHG.csv` | Raw DHG data |
| `8. YH.csv` | Raw YH data |
| `9. HTTL.csv` | Raw HTTL data |
| `10.Climate_reconstruction_summary.py` | Exports regional reconstruction curves from PNG metadata |
| `10.Climate_reconstruction_summary.csv` | Ages and method-specific West/East MAP and MAT reconstructions |

### Processed data

Paths below are relative to `D:\data\Processed data`.

| Path | Contents |
|---|---|
| `relative_abundance\GALORE.py` | Processes raw GALORE data and writes `GALORE_relative_abundance.csv` and `GALORE_presence_absence.csv` in the same directory |
| `relative_abundance\GALORE sift\GALORE sift.py` | Filters data using NLR mappings and the number of valid taxa; writes `GALORE_relative_abundance_sift.csv` |
| `relative_abundance\GALORE sift\distribution.py` | Writes `distribution.png` in the same directory |
| `borehole` | Preprocessing scripts and `KC-1.csv`, `SG-1.csv`, `SG-1b.csv`, `DHG.csv`, `YH.csv`, and `HTTL.csv` |
| `Plant distribution` | Plant distribution data used for reconstruction |
| `Independent_paleoenvironmental_records` | Independent paleoenvironmental comparison records and their `Indicator_summary.csv` |
| `Potential_driving_mechanisms` | Potential driving mechanism data and their `Indicator_summary.csv` |

`borehole\HTTL+YH.py` produces relative-abundance tables from the raw YH and HTTL files, retaining the existing taxonomic mappings and six-decimal precision. Its `Picea+Abies → Picea` mapping is an inherited processing rule; the limitation of this combined taxon should be considered when interpreting results.

## Figure mapping

All scripts are in `D:\code`; PNG outputs default to `D:\results`.

| Script | Output | Contents |
|---|---|---|
| `Fig. 1.py` | `Fig. 1.png` | Borehole locations and topography |
| `Fig. 2.py` | `Fig. 2.png` | Main reconstruction with equal weights for three method groups |
| `Fig. 3.py` | `Fig. 3.png` | Climate and independent paleoenvironmental comparisons |
| `Fig. 4.py` | `Fig. 4.png` | Climate and potential driving mechanism comparisons |
| `Supplementary Fig 1.py` | `Supplementary Fig 1.png` | MAP: Modern Analogue, WA, and WA-PLS |
| `Supplementary Fig 2.py` | `Supplementary Fig 2.png` | MAP: machine learning methods |
| `Supplementary Fig 3.py` | `Supplementary Fig 3.png` | MAP: botanical methods |
| `Supplementary Fig 4.py` | `Supplementary Fig 4.png` | MAT: Modern Analogue, WA, and WA-PLS |
| `Supplementary Fig 5.py` | `Supplementary Fig 5.png` | MAT: machine learning methods |
| `Supplementary Fig 6.py` | `Supplementary Fig 6.png` | MAT: botanical methods |
| `Supplementary Fig 7.py` | `Supplementary Fig 7.png` | Leave-one-method-group-out sensitivity analysis |

MAP means mean annual precipitation; MAT means mean annual temperature. In the main reconstruction, Modern Analogue, WA-PLS, and ML each receive 1/3 of the total weight. RF, ExtraTrees, and Boosting each receive 1/3 within ML.

## Environment and external dependencies

Scripts use local absolute paths. Update script paths when moving directories or changing drive letters.

- Python packages include NumPy, pandas, SciPy, Matplotlib, scikit-learn, Pillow, openpyxl, netCDF4, rasterio, and Cartopy; script imports specify the actual requirements.
- Elevation data are derived from the SRTM digital elevation model (DEM).
- Climate data are from CRU, with a spatial resolution of 0.5°.
- Cartopy maps require the relevant geographic basemap data to be available.

## Running the scripts

Use quoted full paths in PowerShell, for example:

```powershell
python "D:\code\Fig. 1.py"
python "D:\code\Fig. 2.py"
python "D:\code\Fig. 3.py"
python "D:\code\Fig. 4.py"
python "D:\data\10.Climate_reconstruction_summary.py"
```

To reprocess raw data, run `GALORE.py`, then `GALORE sift.py`, followed by the borehole preprocessing scripts. Generate the main reconstruction and supplementary figures before running comparison figures and the summary exporter. `HTTL+YH.py` produces both the YH and HTTL tables.

The summary exporter reads `D:\results` by default and writes `D:\data\10.Climate_reconstruction_summary.csv`. The table contains 716 ages and 45 columns: age plus 11 methods for each of four region/climate-variable groups. Blank values indicate missing temporal support; no extrapolation is performed. The 0.025 Ma display grid does not represent independent temporal resolution.

