import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds, bounds as window_bounds, Window
from rasterio.enums import Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FuncFormatter
csv_path = r"D:\data\4. Borehole_information.csv"
tif_path = r"H:\elevation\wc2.1_30s_elev\wc2.1_30s_elev.tif"
out_path = r"D:\results\Fig. 1.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.linewidth": 1.5,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "axes.unicode_minus": False,
    "savefig.dpi": 600
})
try:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="gb18030")
df.columns = df.columns.str.strip()
df = df[["Name", "Latitude", "Longitude"]].copy()
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Name", "Latitude", "Longitude"])
df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)].reset_index(drop=True)
if df.empty:
    raise ValueError("No valid borehole coordinates found")
lon = df["Longitude"].to_numpy()
lat = df["Latitude"].to_numpy()
main_extent = [
    min(70, float(lon.min()) - 5),
    max(125, float(lon.max()) + 5),
    min(8, float(lat.min()) - 5),
    max(53, float(lat.max()) + 5)
]
local_x_margin = max(2.2, float(np.ptp(lon)) * 0.70 + 0.6)
local_y_margin = max(1.6, float(np.ptp(lat)) * 0.85 + 0.5)
local_extent_requested = [
    float(lon.min()) - local_x_margin,
    float(lon.max()) + local_x_margin,
    float(lat.min()) - local_y_margin,
    float(lat.max()) + local_y_margin
]
edges = np.arange(0, 9001, 1000, dtype=float)
colors = ["#628E4A", "#AEC579", "#F4EA9B", "#F3CE83", "#DEA974", "#BE8577", "#D9A5B6", "#E8D7E2", "#F7F3F7"]
cmap = ListedColormap(colors)
cmap.set_bad("white")
norm = BoundaryNorm(edges, cmap.N, clip=True)
def read_dem(src, extent, max_width=2400, max_height=1800):
    x0, x1, y0, y1 = extent
    x0 = max(x0, src.bounds.left)
    x1 = min(x1, src.bounds.right)
    y0 = max(y0, src.bounds.bottom)
    y1 = min(y1, src.bounds.top)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Map extent does not intersect elevation raster")
    window = from_bounds(x0, y0, x1, y1, src.transform)
    window = window.round_offsets().round_lengths()
    window = window.intersection(Window(0, 0, src.width, src.height))
    factor = min(1.0, max_width / window.width, max_height / window.height)
    w = max(2, int(np.ceil(window.width * factor)))
    h = max(2, int(np.ceil(window.height * factor)))
    data = src.read(
        1,
        window=window,
        out_shape=(h, w),
        resampling=Resampling.bilinear,
        masked=True
    ).astype(np.float32)
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where((data < -500) | (data > 9000), data)
    data = np.ma.maximum(data, 0)
    left, bottom, right, top = window_bounds(window, src.transform)
    return data, [left, right, bottom, top]
def add_terrain(ax, data, extent, shade_alpha=0.12):
    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        zorder=1
    )
    valid = data.compressed()
    if valid.size:
        surface = data.filled(float(np.median(valid)))
        dx = abs(extent[1] - extent[0]) * 111320 * np.cos(np.deg2rad((extent[2] + extent[3]) / 2)) / data.shape[1]
        dy = abs(extent[3] - extent[2]) * 111320 / data.shape[0]
        shade = LightSource(azdeg=315, altdeg=45).hillshade(
            surface,
            vert_exag=2,
            dx=max(dx, 1),
            dy=max(dy, 1)
        )
        shade = np.ma.array(shade, mask=np.ma.getmaskarray(data))
        ax.imshow(
            shade,
            extent=extent,
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=shade_alpha,
            interpolation="bilinear",
            aspect="auto",
            zorder=2
        )
    return im
def format_map(ax, extent, xstep, ystep, inset=False):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("auto")
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{abs(v):g}°{'E' if v >= 0 else 'W'}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{abs(v):g}°{'N' if v >= 0 else 'S'}"))
    ax.xaxis.set_ticks_position("top" if inset else "bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=7 if inset else 10,
        width=1.4 if inset else 1.6,
        pad=5 if inset else 8,
        labelsize=17 if inset else 21,
        top=inset,
        bottom=not inset,
        left=True,
        right=False,
        labeltop=inset,
        labelbottom=not inset,
        labelleft=True,
        labelright=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
        spine.set_color("black")
        spine.set_zorder(25)
def add_elevation_legend(fig, im):
    panel = fig.add_axes([0.682, 0.785, 0.265, 0.135], facecolor="white", zorder=30)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#555555")
    panel.text(0.5, 0.80, "Elevation (m)", ha="center", va="center", fontsize=19, fontfamily="Times New Roman", color="black")
    cax = panel.inset_axes([0.085, 0.35, 0.83, 0.24])
    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        boundaries=edges,
        ticks=[0, 2000, 4000, 6000, 8000],
        spacing="uniform",
        drawedges=True
    )
    cbar.ax.tick_params(axis="x", labelsize=15, length=4, width=1.0, pad=4)
    cbar.outline.set_linewidth(0.9)
    cbar.solids.set_edgecolor("#777777")
    cbar.solids.set_linewidth(0.45)
def normalize_name(name):
    return re.sub(r"\s+", "", str(name).upper()).replace("–", "-").replace("—", "-")
def add_borehole_labels(ax, points):
    fixed = {"KC-1": (-19, -17), "SG-1B": (-19, 18), "SG-1": (19, -17)}
    candidate_offsets = [
        (17, 15), (-17, 15), (17, -15), (-17, -15),
        (22, 3), (-22, 3), (11, 23), (-11, 23),
        (11, -23), (-11, -23), (25, 12), (-25, 12)
    ]
    entries = []
    for _, row in points.iterrows():
        entries.append((str(row["Name"]), normalize_name(row["Name"]), float(row["Longitude"]), float(row["Latitude"])))
    entries.sort(key=lambda item: 0 if item[1] in fixed else 1)
    annotations = []
    fig = ax.figure
    fig.canvas.draw()
    axes_box = ax.get_window_extent()
    point_pixels = ax.transData.transform(np.column_stack([points["Longitude"].to_numpy(), points["Latitude"].to_numpy()]))
    def create_annotation(name, x, y, ox, oy):
        ha = "left" if ox > 0 else "right"
        va = "bottom" if oy > 5 else "top" if oy < -5 else "center"
        return ax.annotate(
            name,
            xy=(x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=17,
            fontfamily="Times New Roman",
            ha=ha,
            va=va,
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92),
            arrowprops=dict(
                arrowstyle="->",
                color="#222222",
                lw=0.85,
                mutation_scale=10,
                shrinkA=1,
                shrinkB=6,
                connectionstyle="arc3,rad=0"
            ),
            zorder=21,
            annotation_clip=True
        )
    for name, key, x, y in entries:
        candidates = [fixed[key]] if key in fixed else candidate_offsets
        best_offset = candidates[0]
        best_score = float("inf")
        for ox, oy in candidates:
            ann = create_annotation(name, x, y, ox, oy)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = ann.get_bbox_patch().get_window_extent(renderer)
            score = np.hypot(ox, oy)
            if bbox.x0 < axes_box.x0 + 8 or bbox.x1 > axes_box.x1 - 8 or bbox.y0 < axes_box.y0 + 8 or bbox.y1 > axes_box.y1 - 8:
                score += 50000
            expanded = bbox.expanded(1.10, 1.20)
            for px, py in point_pixels:
                if expanded.x0 - 6 <= px <= expanded.x1 + 6 and expanded.y0 - 6 <= py <= expanded.y1 + 6:
                    score += 10000
            for previous in annotations:
                prev_box = previous.get_bbox_patch().get_window_extent(renderer).expanded(1.06, 1.10)
                if expanded.overlaps(prev_box):
                    score += 30000
            if score < best_score:
                best_score = score
                best_offset = (ox, oy)
            ann.remove()
        annotations.append(create_annotation(name, x, y, *best_offset))
    return annotations
with rasterio.open(tif_path) as src:
    if src.crs is None or src.crs.to_epsg() != 4326:
        raise ValueError("WorldClim elevation raster must use EPSG:4326")
    main_dem, main_bounds = read_dem(src, main_extent, 2600, 2000)
    local_dem, local_plot_extent = read_dem(src, local_extent_requested, 2300, 1500)
fig = plt.figure(figsize=(15, 11.5), facecolor="white")
ax = fig.add_axes([0.065, 0.12, 0.89, 0.81])
im = add_terrain(ax, main_dem, main_bounds, 0.11)
format_map(ax, main_extent, 10, 10, inset=False)
ax.scatter(lon, lat, s=120, c="#F21D1D", edgecolors="black", linewidths=0.8, zorder=8)
rx0, rx1, ry0, ry1 = local_plot_extent
rect = Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, linewidth=2.5, edgecolor="#E11A1A", facecolor="none", zorder=10)
ax.add_patch(rect)
inset = fig.add_axes([0.39, 0.12, 0.565, 0.36], facecolor="white", zorder=15)
add_terrain(inset, local_dem, local_plot_extent, 0.12)
xstep = 1 if local_plot_extent[1] - local_plot_extent[0] <= 10 else 2
ystep = 1 if local_plot_extent[3] - local_plot_extent[2] <= 7 else 2
format_map(inset, local_plot_extent, xstep, ystep, inset=True)
inset.scatter(lon, lat, s=155, c="#F21D1D", edgecolors="black", linewidths=0.9, zorder=18)
add_borehole_labels(inset, df)
add_elevation_legend(fig, im)
fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="white")
plt.close(fig)