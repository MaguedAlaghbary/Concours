import streamlit as st
import folium
from streamlit_folium import st_folium
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize, LogNorm
from io import BytesIO
import base64
import pickle

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Djibouti Aquifer Vulnerability", layout="wide")
st.title("🗺️ Djibouti Nitrate Vulnerability Mapper")
st.markdown("**DRASTICLU + ML-based assessment with full prediction analysis**")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_resource
def load_data():
    with open('djibouti_data_minimal.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data_xr = load_data()
    st.success("✅ Data loaded")
except FileNotFoundError:
    st.error("❌ Missing 'djibouti_data_minimal.pkl'")
    st.stop()

# ============================================================================
# LOAD NITRATE MEASUREMENT POINTS
# ============================================================================
@st.cache_data
def load_nitrate_points():
    try:
        df_nitrate = pd.read_csv('lat_lon_d_n_data.csv')
        # Ensure column names are correct
        df_nitrate.columns = df_nitrate.columns.str.strip().str.lower()
        return df_nitrate
    except FileNotFoundError:
        st.warning("⚠️ Nitrate measurements file not found: lat_lon_d_n_data.csv")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading nitrate data: {str(e)}")
        return None

df_nitrate_points = load_nitrate_points()

# ============================================================================
# FUNCTION TO ADD NITRATE POINTS TO FOLIUM MAP
# ============================================================================
def add_nitrate_layer(m, df_nitrate, cmap, norm_obj, show_points=True):
    """Add nitrate measurement points as a folium FeatureGroup (toggleable)"""
    if df_nitrate is None or not show_points:
        return m
    
    fg_nitrate = folium.FeatureGroup(name='🧪 Nitrate Measurements (mg/L)', show=True)
    
    # Get colormap RGBA values
    for idx, row in df_nitrate.iterrows():
        try:
            lon = float(row['longitude'])
            lat = float(row['latitude'])
            no3_val = float(row['no3'])  # or 'NO3' or 'nitrate' depending on column name
            
            # Normalize and get color
            normalized_val = norm_obj(no3_val)
            rgba = cmap(normalized_val)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0]*255), 
                int(rgba[1]*255), 
                int(rgba[2]*255)
            )
            
            # Add circle marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=f"NO₃⁻: {no3_val:.1f} mg/L<br>Lat: {lat:.4f}°<br>Lon: {lon:.4f}°",
                color=hex_color,
                fill=True,
                fillColor=hex_color,
                fillOpacity=0.8,
                weight=1,
                opacity=0.9
            ).add_to(fg_nitrate)
        except (ValueError, KeyError) as e:
            continue
    
    fg_nitrate.add_to(m)
    return m

# ============================================================================
# DEFINE COLOR SCHEMES (EXACT from Douda notebook)
# ============================================================================

# ============================================================================
# CATEGORY LABEL MAPS (For categorical input layers)
# ============================================================================

LITHOLOGY_NAME_MAP = {
    0: "NoData",
    1: "Dalha Basalts",
    2: "Mablas Acidic Series",
    3: "Stratoid Basalts",
    4: "Quaternary Sediments",
    5: "Water",
    6: "Gulf Basalts",
    7: "Axial Series",
    8: "Ali Sabieh Basalts",
    9: "Somali Basalts",
    10: "Mesozoic Formation",
    -1: "Merged"
}

SOIL_TEXTURE_MAPPING = {
    0: "No Data",
    1: "Clay",
    2: "Silty Clay", 
    3: "Sandy Clay",
    4: "Clay Loam",
    5: "Silty Clay Loam",
    6: "Sandy Clay Loam",
    7: "Loam",
    8: "Silty Loam",
    9: "Sandy Loam",
    10: "Silt",
    11: "Loamy Sand",
    12: "Sand",
    -1: "Merged"
}

LANDCOVER_LABEL_MAP = {
    0: "NoData",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare",
    70: "Snow/Ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/Lichen",
    -1: "Merged"
}

# 9-level risk colormap - Blue-White-Red diverging
risk_9_colors = {
    1: '#053061',  # Very Low - Dark blue
    2: '#2166AC',  # Low - Blue
    3: '#4393C3',  # Low-Moderate - Light blue
    4: '#cceeff',  # Moderate-Low - Pale blue
    5: '#F7F7F7',  # Moderate - White/neutral
    6: '#F4A582',  # Moderate-High - Pale red
    7: '#D6604D',  # High-Moderate - Light red
    8: '#B2182B',  # High - Red
    9: '#67001F',  # Very High - Dark red
}

# 4-level priority colormap
priority_4_colors = {
    1: '#2166AC',  # Low Risk - Dark blue
    2: '#cceeff',  # Moderate Risk - Light blue
    3: '#F4A582',  # High Risk - Light red
    4: '#B2182B',  # Very High Risk - Dark red
}

# NOTE: cmap/norm for every class layer (Risk, Priority, defuzzified classes,
# DRASTICLU categorical inputs, driver rank/SHAP) are now derived automatically
# inside plot_class_layer() from the *_colors dicts above + the codes actually
# present in the data. This removes a prior off-by-one BoundaryNorm bug (the
# top class in Risk/vulnerability/nitrate never got its own color bin) and a
# separate misalignment bug for non-contiguous codes (e.g. land-cover 10..100).

# ============================================================================
# LABELS
# ============================================================================
RISK_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Low-Moderate",
    4: "Moderate-Low",
    5: "Moderate",
    6: "Moderate-High",
    7: "High-Moderate",
    8: "High",
    9: "Very High"
}

PRIORITY_LABELS = {
    1: "Low Risk",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Very High Risk"
}

DRASTIC_LABELS = {
    'D': 'Depth to Water', 'R': 'Recharge Rate', 'A': 'Aquifer Media',
    'S': 'Soil Type', 'T': 'Topography', 'I': 'Impact Vadose Zone',
    'C': 'Conductivity', 'LU': 'Land Use'
}

DRIVER_MAP = {
    0: 'D (Depth)', 1: 'R (Recharge)', 2: 'A (Aquifer)', 3: 'S (Soil)',
    4: 'T (Topography)', 5: 'I (Impact)', 6: 'C (Conductivity)', 7: 'LU (Land Use)'
}

# ============================================================================
# PREDICTION LAYER TITLES & COLORMAPS (FROM NOTEBOOK)
# ============================================================================

# ============================================================================
# PREDICTION LAYER TITLES & COLORMAPS (EXACT FROM NOTEBOOK)
# ============================================================================

PREDICTION_TITLES = {
    'index_shap': "Specific Vulnerability",
    'index_shap_std': "Specific Vulnerability: Uncertainty",
    'index_shap_class': "Defuzzified Specific Vulnerability",
    'index_shap_entropy_norm': "Defuzzified Specific Vulnerability: Uncertainty",
    'y_hat': "NO₃⁻ Concentrations",
    'y_hat_std': "NO₃⁻ Concentrations: Uncertainty",
    'y_hat_log_class': "Defuzzified NO₃⁻ Contamination",
    'y_hat_log_entropy_norm': "Defuzzified NO₃⁻ Contamination: Uncertainty",
}

# Continuous colormaps (EXACT from notebook)
cmap_nitrate = mcolors.LinearSegmentedColormap.from_list(
    'nitrate_contamination',
    ['#FFEDA0', '#FED976', '#FEB24C', '#F03B20', '#BD0026']
)

cmap_vulnerability = mcolors.LinearSegmentedColormap.from_list(
    'vulnerability_index',
    ['#440154', '#31688E', '#35B779', '#FDE724', '#CC4C02']
)

# Use viridis for uncertainty (std dev)
cmap_std = plt.cm.viridis

# Use davos-inspired for entropy - teal colormap (low entropy = light, high = dark)
cmap_entropy = mcolors.LinearSegmentedColormap.from_list(
    'davos',
    ['#F0FFFF', '#A7D8DE', '#5A9FA5', '#2F5F66', '#0D2626']
)

# Use davos for entropy (from cmocean)
# ============================================================================
# ENTROPY COLORMAP (DAVOS or fallback - LIGHT SEQUENTIAL)
# ============================================================================
try:
    import cmocean.cm as cmo
    cmap_entropy = cmo.davos
except ImportError:
    # Fallback: light blue-cyan sequential (davos-inspired, lighter)
    cmap_entropy = mcolors.LinearSegmentedColormap.from_list(
        'davos_light',
        ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5']
    )

# 5-class categorical for defuzzified layers (with labels)
# 5-class categorical colormaps (for defuzzified)
# ============================================================================
# CLASS LABELS FOR DEFUZZIFIED MAPS
# ============================================================================

# ============================================================================
# CLASS LABELS FOR DEFUZZIFIED MAPS (WITH RANGES)
# ============================================================================

vulnerability_class_labels = {
    1: "Very Low (≤100)",
    2: "Low (100-136)",
    3: "Moderate (136-166)",
    4: "High (166-174)",
    5: "Very High (≥174)"
}

nitrate_class_labels = {
    1: "Very Low (≤10 mg/L)",
    2: "Low (10-25 mg/L)",
    3: "Moderate (25-50 mg/L)",
    4: "High (50-100 mg/L)",
    5: "Very High (≥100 mg/L)"
}

# 5-class categorical colormaps (for defuzzified)
vulnerability_5_colors = {
    1: '#440154',  # Very Low - Dark purple
    2: '#31688E',  # Low - Blue
    3: '#35B779',  # Moderate - Green
    4: '#FDE724',  # High - Yellow
    5: '#CC4C02',  # Very High - Dark orange
}

nitrate_5_colors = {
    1: '#FFEDA0',  # Very Low - Light yellow
    2: '#FED976',  # Low - Yellow
    3: '#FEB24C',  # Moderate - Orange
    4: '#F03B20',  # High - Red-orange
    5: '#BD0026',  # Very High - Dark red
}

# ============================================================================
# PARAMETER COLORS - Paul Tol's "Bright" scheme (8 DRASTICLU parameters)
# ============================================================================
parameters_8_colors = {
    1: '#4477AA',  # Blue - D (Depth to water)
    2: '#EE6677',  # Red - R (Recharge)
    3: '#228833',  # Green - A (Aquifer media)
    4: '#CCBB44',  # Yellow - S (Soil media)
    5: '#B2DF8A',  # Light green - T (Topography)
    6: '#AA3377',  # Purple - I (Impact of vadose)
    7: '#BBBBBB',  # Grey - C (Conductivity)
    8: '#EE99AA',  # Pink - LU (Land use)
}

# Map driver indices to parameter codes
DRIVER_PARAM_MAP = {
    0: 'D',
    1: 'R',
    2: 'A',
    3: 'S',
    4: 'T',
    5: 'I',
    6: 'C',
    7: 'LU'
}


# ============================================================================
# INPUT LAYERS CONFIGURATION (DRASTICLU)
# ============================================================================

INPUT_LAYERS_CONFIG = [
    {
        "layer": "D",
        "title": "Depth to Water Table",
        "units": "[m]",
        "cmap": "viridis",
        "vmin": None,  # Will calculate from data
        "vmax": None,
        "quantile_min": 0.05,
        "quantile_max": 0.95
    },
    {
        "layer": "R",
        "title": "Recharge",
        "units": "[mm yr⁻¹]",
        "cmap": "YlGn",
        "vmin": 10,
        "vmax": None,
        "quantile_max": 0.75
    },
    {
        "layer": "A",
        "title": "Aquifer Media",
        "units": "[Lithology]",
        "categorical": True,
        "legend": LITHOLOGY_NAME_MAP,
        "colors": {0: "#a6cee3", 1: "#1f78b4", 2: "#b2df8a", 3: "#33a02c",
                   4: "#fb9a99", 5: "#e31a1c", 6: "#fdbf6f", 7: "#ff7f00",
                   8: "#cab2d6", 9: "#6a3d9a", 10: "#b15928"}
    },
    {
        "layer": "S",
        "title": "Soil Media",
        "units": "[Soil Texture]",
        "categorical": True,
        "legend": SOIL_TEXTURE_MAPPING,
        "colors": {0: "#66c2a5", 1: "#fc8d62", 2: "#a6d854", 3: "#8da0cb",
                   4: "#e78ac3", 5: "#d9d9d9", 6: "#ffd92f", 7: "#e5c494",
                   8: "#b3b3b3", 9: "#bc80bd", 10: "#fbbf9b", 11: "#cad5d0", 12: "#f4a582"}
    },
    {
        "layer": "T",
        "title": "Topography",
        "units": "[%]",
        "cmap": "cividis",
        "vmin": 3,
        "vmax": None,
        "quantile_max": 0.95
    },
    {
        "layer": "I",
        "title": "Impact of Vadose Zone",
        "units": "[d⁻¹]",
        "cmap": "RdYlGn",
        "vmin": None,
        "vmax": None,
        "quantile_min": 0.05,
        "quantile_max": 0.95
    },
    {
        "layer": "C",
        "title": "Hydraulic Conductivity",
        "units": "[m d⁻¹]",
        "cmap": "magma",
        "vmin": 0.05,
        "vmax": 95,
        "log_scale": True
    },
    {
        "layer": "LU",
        "title": "Land Use",
        "units": "[Land Cover]",
        "categorical": True,
        "legend": LANDCOVER_LABEL_MAP,
        "colors": {10: "#66c2a5", 20: "#a6d854", 30: "#ffd92f", 40: "#e78ac3",
                   50: "#fc8d62", 60: "#8da0cb", 70: "#e5c494", 80: "#b3b3b3",
                   90: "#ffffbf", 95: "#1b9e77", 100: "#d95f02", -1: "#cccccc"}
    }
]



# ============================================================================
# SIDEBAR: LOCATION INPUT
# ============================================================================
st.sidebar.header("📍 Query Location")

col1, col2 = st.sidebar.columns(2)
with col1:
    lat_input = st.number_input("Latitude", min_value=10.9, max_value=12.7, value=11.5, step=0.01, key="lat_slider")
with col2:
    lon_input = st.number_input("Longitude", min_value=41.7, max_value=43.4, value=42.9, step=0.01, key="lon_slider")

st.sidebar.info(f"**Selected:** {lat_input:.3f}°N, {lon_input:.3f}°E")

# ============================================================================
# FUNCTION: Extract values at point
# ============================================================================
def extract_at_point(lat, lon, data_xr, vars_list):
    results = {}
    for var in vars_list:
        try:
            val = float(data_xr[var].sel(latitude=lat, longitude=lon, method='nearest').values)
            results[var] = val
        except:
            try:
                val = float(data_xr[var].sel(lat=lat, lon=lon, method='nearest').values)
                results[var] = val
            except:
                results[var] = np.nan
    return results

# ============================================================================
# SHARED RENDERING HELPERS (used by both plotters below)
# ============================================================================
def _get_water_mask(data_xr):
    """Boolean mask of water-covered pixels (LU == 80), blanked out on every map."""
    return data_xr['LU'].values == 80


def _get_domain_bounds(data_xr):
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    return lats.min(), lats.max(), lons.min(), lons.max()


def _fig_to_base64(fig, **savefig_kwargs):
    """Render a matplotlib figure to a base64 PNG string and close it."""
    buf = BytesIO()
    fig.savefig(buf, format='png', **savefig_kwargs)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _render_raster_png(raster_2d, water_mask, lon_min, lon_max, lat_min, lat_max,
                        cmap, norm, figsize=(8, 8), dpi=100):
    """Draw a masked raster with no axes/padding, return (base64_png, masked_array)."""
    raster_masked = np.ma.masked_where(water_mask, raster_2d)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='none')
    fig.patch.set_alpha(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0)
    ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
              cmap=cmap, norm=norm, origin='lower', alpha=0.9, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    img_b64 = _fig_to_base64(fig, bbox_inches='tight', dpi=dpi, facecolor='none',
                              edgecolor='none', transparent=True, pad_inches=0)
    return img_b64, raster_masked


def _make_base_folium_map(lat_min, lat_max, lon_min, lon_max, img_b64,
                           lat, lon, marker_color, popup_text, marker_radius=6):
    """Folium map + raster ImageOverlay + a CircleMarker at the queried point."""
    m = folium.Map(
        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_b64}",
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.85,
        interactive=True,
        cross_origin=False
    ).add_to(m)
    folium.CircleMarker(
        location=[lat, lon],
        radius=marker_radius,
        popup=popup_text,
        color=marker_color,
        fill=True,
        fillColor=marker_color,
        fillOpacity=0.95,
        weight=2
    ).add_to(m)
    return m


def _class_cmap_norm(class_colors, present_values):
    """
    Build a ListedColormap + BoundaryNorm that gives each ACTUAL present code its
    own bin, using midpoints between the sorted present codes as bin edges.

    This replaces two bugs in the previous hand-rolled BoundaryNorm calls:
      1. Off-by-one boundary count (e.g. BoundaryNorm(np.arange(0.5, 9.5, 1), 9) for
         Risk only defines 8 bins for 9 classes, so class 9 "Very High" silently
         reused class 8's color). Building boundaries from the data guarantees the
         correct count every time.
      2. Position/value mismatch for non-contiguous codes (e.g. land-cover codes
         10, 20, ..., 100): a fixed-step norm assuming codes are 0..n-1 puts almost
         every pixel in the same "overflow" bin. Using the real codes' midpoints
         fixes this regardless of spacing or sign (handles -1 "Merged" codes too).
    """
    present = sorted(present_values)
    colors = [class_colors.get(v, '#cccccc') for v in present]
    cmap = ListedColormap(colors)
    if len(present) == 1:
        boundaries = [present[0] - 0.5, present[0] + 0.5]
    else:
        mids = [(a + b) / 2 for a, b in zip(present[:-1], present[1:])]
        lo = present[0] - (mids[0] - present[0])
        hi = present[-1] + (present[-1] - mids[-1])
        boundaries = [lo] + mids + [hi]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, present


# ============================================================================
# PLOTTER 1 of 2 — CONTINUOUS layers: raster overlay + a colorbar SCALE
# ============================================================================
def plot_continuous_layer(data_xr, var_name, cmap, norm, title, lat, lon,
                           units="", water_mask=None, marker_color='red', figsize=(8, 8)):
    """
    Render a continuous (float-valued) raster layer as a folium map.

    cmap/norm are supplied by the CALLER (e.g. Normalize(vmin=..., vmax=...) or
    LogNorm(...)) so every value range chosen per-variable is preserved exactly.
    The legend rendered is a colorbar SCALE — no discrete class list.
    """
    try:
        raster_data = data_xr[var_name].values
    except KeyError:
        return None

    if water_mask is None:
        water_mask = _get_water_mask(data_xr)
    lat_min, lat_max, lon_min, lon_max = _get_domain_bounds(data_xr)

    img_b64, _ = _render_raster_png(raster_data, water_mask, lon_min, lon_max,
                                     lat_min, lat_max, cmap, norm, figsize=figsize)

    # Small horizontal colorbar image = the "scale" legend for this layer
    cbar_fig, cbar_ax = plt.subplots(figsize=(1.8, 0.25), dpi=80)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cbar_ax, orientation='horizontal', pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    cbar_b64 = _fig_to_base64(cbar_fig, bbox_inches='tight', dpi=80,
                               facecolor='white', transparent=False, pad_inches=0.02)

    try:
        point_val = float(data_xr[var_name].sel(latitude=lat, longitude=lon, method='nearest').values)
        value_str = f"{point_val:.3f}{(' ' + units) if units else ''}"
    except Exception:
        value_str = "N/A"

    popup_text = f"<b>{title}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Value: {value_str}"
    m = _make_base_folium_map(lat_min, lat_max, lon_min, lon_max, img_b64, lat, lon,
                               marker_color, popup_text)

    legend_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background-color: white; border:2px solid #333; z-index:9999;
                border-radius: 3px; padding: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
    <div style="font-size: 10px; font-weight: bold; text-align: center; margin-bottom: 4px;">{title}</div>
    <img src="data:image/png;base64,{cbar_b64}" style="width: 160px; height: auto; display: block; margin: 0 auto;">
    </div>'''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ============================================================================
# PLOTTER 2 of 2 — CLASS layers: raster overlay + a swatch+label LEGEND
# ============================================================================
def plot_class_layer(data_xr, var_name, class_colors, class_labels, title, lat, lon,
                      water_mask=None, figsize=(8, 8), show_legend=True):
    """
    Render a categorical (integer-coded) raster layer as a folium map.

    class_colors/class_labels map the RAW codes stored in data_xr[var_name] to a
    color and a human-readable name. cmap/norm are DERIVED here (not passed in)
    via _class_cmap_norm so every class is guaranteed a correctly-bounded bin —
    see _class_cmap_norm's docstring for the two bugs this avoids. The legend
    lists every class actually present in the layer, by name (not a colorbar).
    """
    try:
        raster_data = data_xr[var_name].values
    except KeyError:
        return None

    if water_mask is None:
        water_mask = _get_water_mask(data_xr)
    lat_min, lat_max, lon_min, lon_max = _get_domain_bounds(data_xr)

    present_preview = np.ma.masked_where(water_mask, raster_data)
    present_values = np.unique(present_preview.compressed()).astype(int).tolist()
    if not present_values:
        present_values = [0]
    cmap, norm, present = _class_cmap_norm(class_colors, present_values)

    img_b64, _ = _render_raster_png(raster_data, water_mask, lon_min, lon_max,
                                     lat_min, lat_max, cmap, norm, figsize=figsize)

    try:
        raw_val = data_xr[var_name].sel(latitude=lat, longitude=lon, method='nearest').values
        selected_value = int(float(raw_val))
        label = class_labels.get(selected_value, str(selected_value))
    except Exception:
        selected_value = None
        label = "N/A"
    marker_color = class_colors.get(selected_value, 'red')
    popup_text = f"<b>{title}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Class: {selected_value} ({label})"

    m = _make_base_folium_map(lat_min, lat_max, lon_min, lon_max, img_b64, lat, lon,
                               marker_color, popup_text, marker_radius=4)

    if show_legend:
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 170px;
                    background-color: white; border:2px solid grey; z-index:9999; font-size:9px;
                    border-radius: 5px; padding: 8px; font-weight: bold;">
        {title}<br>'''
        for k in present:
            lbl = class_labels.get(k, str(k))
            color = class_colors.get(k, '#cccccc')
            legend_html += (f'<div style="margin: 2px 0; font-weight: normal;">'
                             f'<i style="background:{color}; width: 12px; height: 12px; '
                             f'float: left; margin-right: 5px; border-radius: 1px; '
                             f'display: inline-block;"></i>{lbl}</div>')
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ============================================================================
# TAB STRUCTURE
# ============================================================================

tab_inputs, tab1, tab2, tab3 = st.tabs([
    "📥 DRASTICLU Inputs",
    "📈 Prediction Maps",
    "🗺️ Risk & Priority",
    "🎯 Attributions", 
])

# ============================================================================
# TAB 0: DRASTICLU INPUT LAYERS
# ============================================================================
# ============================================================================
# TAB 0: DRASTICLU INPUT LAYERS
# ============================================================================
# ============================================================================
# TAB 0: DRASTICLU INPUT LAYERS
# ============================================================================
with tab_inputs:
    st.header("📥 DRASTICLU Input Layers (8 Parameters)")
    
    # Layer selector (centered)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        layer_names = [f"{c['layer']} — {c['title']}" for c in INPUT_LAYERS_CONFIG]
        selected_idx = st.selectbox("Choose a layer:", range(len(INPUT_LAYERS_CONFIG)), 
                                      format_func=lambda i: layer_names[i], key="layer_select")
        config = INPUT_LAYERS_CONFIG[selected_idx]
        st.info(f"**{config['title']}** | {config['units']}")
    
    # Nitrate points toggle (only relevant for y_hat, but offered for convenience)
    show_nitrate_points = st.checkbox("🧪 Show Nitrate Measurement Points", value=False, key="show_nitrate_toggle_inputs")
    
    if config['layer'] not in data_xr:
        st.error(f"❌ Layer {config['layer']} not found in data")
        st.stop()
    
    water_mask = _get_water_mask(data_xr)
    
    # Render the chosen layer
    if config.get('categorical'):
        # CLASS plotter
        m = plot_class_layer(
            data_xr, config['layer'],
            class_colors=config['colors'], class_labels=config['legend'],
            title=f"{config['title']} {config['units']}",
            lat=lat_input, lon=lon_input, water_mask=water_mask, figsize=(8, 8)
        )
    else:
        # CONTINUOUS plotter: resolve vmin/vmax (same logic as before)
        layer_masked = np.ma.masked_where(water_mask, data_xr[config['layer']].values)
        vmin, vmax = config.get('vmin'), config.get('vmax')
        
        if vmin is None or vmax is None:
            valid_data = layer_masked.compressed()
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.quantile(valid_data, config['quantile_min']) if 'quantile_min' in config else float(valid_data.min())
                if vmax is None:
                    vmax = np.quantile(valid_data, config['quantile_max']) if 'quantile_max' in config else float(valid_data.max())
            else:
                vmin, vmax = vmin or 0, vmax or 1
        
        if vmin is None or vmax is None or vmin >= vmax:
            valid_data = layer_masked.compressed()
            if len(valid_data) > 0:
                vmin, vmax = float(np.nanmin(valid_data)), float(np.nanmax(valid_data))
            else:
                vmin, vmax = 0, 1
        
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        
        norm_cont = (LogNorm(vmin=max(vmin, 0.01), vmax=vmax) if config.get('log_scale')
                     else Normalize(vmin=vmin, vmax=vmax))
        
        m = plot_continuous_layer(
            data_xr, config['layer'], cmap=config['cmap'], norm=norm_cont,
            title=f"{config['title']} {config['units']}", units=config['units'],
            lat=lat_input, lon=lon_input, water_mask=water_mask, figsize=(8, 8)
        )
    
    # Add nitrate measurement points ONLY if:
    # (1) checkbox is checked, (2) we're viewing the y_hat (nitrate concentration) layer, (3) map rendered successfully
    if show_nitrate_points and config['layer'] == 'y_hat' and m is not None and df_nitrate_points is not None:
        norm_yhat = Normalize(vmin=10, vmax=100)
        m = add_nitrate_layer(m, df_nitrate_points, cmap_nitrate, norm_yhat, True)
    
    # Render map centered on page
    if m:
        col_l, col_c, col_r = st.columns([1, 1.5, 1])
        with col_c:
            st_folium(m, width=600, height=500, key=f"layer_{config['layer']}_{lat_input}_{lon_input}")

# ============================================================================
# TAB 1: RISK & PRIORITY MAPS
# ============================================================================
with tab3:
    st.header("Risk & Priority Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contamination Risk (DRASTICLU, 1-9)")
        risk_map = plot_class_layer(
            data_xr, 'risk_pdp_shap',
            class_colors=risk_9_colors, class_labels=RISK_LABELS,
            title="Risk", lat=lat_input, lon=lon_input
        )
        if risk_map:
            st_folium(risk_map, width=350, height=350, key=f"risk_map_{lat_input}_{lon_input}")
    
    with col2:
        st.subheader("Management Priority (1-4)")
        priority_map = plot_class_layer(
            data_xr, 'priority_zones_regulatory',
            class_colors=priority_4_colors, class_labels=PRIORITY_LABELS,
            title="Priority", lat=lat_input, lon=lon_input
        )
        if priority_map:
            st_folium(priority_map, width=350, height=350, key=f"priority_map_{lat_input}_{lon_input}")
# ============================================================================
# TAB 2: RIVER SHAP ATTRIBUTION MAPS
# ============================================================================
with tab2:
    st.header("🎯 Driver Attribution Analysis (Rank & SHAP)")
    
    st.info("Top: Driver Rank (1-4) | Bottom: Driver SHAP values (1-4)")
    
    # Verify driver layers exist (fail fast with one clear message)
    try:
        for i in range(1, 5):
            _ = data_xr[f'driver_rank_{i}']
            _ = data_xr[f'driver_shap_{i}']
    except KeyError:
        st.error("Cannot load driver data")
        st.stop()
    
    water_mask = _get_water_mask(data_xr)
    
    # Driver colors
    driver_colors = {
        0: parameters_8_colors[1],  # D
        1: parameters_8_colors[2],  # R
        2: parameters_8_colors[3],  # A
        3: parameters_8_colors[4],  # S
        4: parameters_8_colors[5],  # T
        5: parameters_8_colors[6],  # I
        6: parameters_8_colors[7],  # C
        7: parameters_8_colors[8],  # LU
    }
    
    # ====== SECTION 1: DRIVER RANK (1-4) ======
    st.subheader("📊 Driver Rank (Most Influential)")
    
    col_map1, col_map2, col_legend1 = st.columns([1, 1, 0.6])
    
    ranks_to_plot = [1, 2, 3, 4]
    rank_positions = [
        (col_map1, 1), (col_map2, 2),
        (col_map1, 3), (col_map2, 4)
    ]
    
    for rank, (col, pos) in zip(ranks_to_plot, rank_positions):
        with col:
            st.text(f"Rank {rank}", help=f"Driver ranking position {rank}")
            # CLASS plotter (which DRASTICLU parameter dominates here) - shared
            # legend column below, so the per-map legend is suppressed
            m = plot_class_layer(
                data_xr, f'driver_rank_{rank}',
                class_colors=driver_colors, class_labels=DRIVER_MAP,
                title=f"Rank {rank}", lat=lat_input, lon=lon_input,
                water_mask=water_mask, figsize=(7, 7), show_legend=False
            )
            if m:
                st_folium(m, width=300, height=300, key=f"driver_rank_{rank}_{lat_input}_{lon_input}")
    
    with col_legend1:
        st.subheader("🎨 Legend", help="Parameter colors")
        param_list = [
            ('D', 'Depth', 1),
            ('R', 'Recharge', 2),
            ('A', 'Aquifer', 3),
            ('S', 'Soil', 4),
            ('T', 'Topography', 5),
            ('I', 'Impact', 6),
            ('C', 'Conductivity', 7),
            ('LU', 'Land Use', 8),
        ]
        
        for code, name, param_num in param_list:
            color = parameters_8_colors[param_num]
            st.markdown(
                f'<div style="padding: 5px; background-color: {color}; color: white; border-radius: 2px; margin: 2px 0; font-size: 11px;">'
                f'<b>{code}</b> {name}</div>',
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # ====== SECTION 2: DRIVER SHAP (1-4) ======
    st.subheader("🔍 Driver SHAP Contribution (Top 4)")
    
    col_map3, col_map4, col_legend2 = st.columns([1, 1, 0.6])
    
    shap_ranks_to_plot = [1, 2, 3, 4]
    shap_positions = [
        (col_map3, 1), (col_map4, 2),
        (col_map3, 3), (col_map4, 4)
    ]
    
    for shap_rank, (col, pos) in zip(shap_ranks_to_plot, shap_positions):
        with col:
            st.text(f"SHAP Rank {shap_rank}", help=f"SHAP contribution ranking {shap_rank}")
            m = plot_class_layer(
                data_xr, f'driver_shap_{shap_rank}',
                class_colors=driver_colors, class_labels=DRIVER_MAP,
                title=f"SHAP Rank {shap_rank}", lat=lat_input, lon=lon_input,
                water_mask=water_mask, figsize=(7, 7), show_legend=False
            )
            if m:
                st_folium(m, width=300, height=300, key=f"driver_shap_{shap_rank}_{lat_input}_{lon_input}")
    
    with col_legend2:
        st.subheader("🎨 Legend", help="Parameter colors")
        param_list = [
            ('D', 'Depth', 1),
            ('R', 'Recharge', 2),
            ('A', 'Aquifer', 3),
            ('S', 'Soil', 4),
            ('T', 'Topography', 5),
            ('I', 'Impact', 6),
            ('C', 'Conductivity', 7),
            ('LU', 'Land Use', 8),
        ]
        
        for code, name, param_num in param_list:
            color = parameters_8_colors[param_num]
            st.markdown(
                f'<div style="padding: 5px; background-color: {color}; color: white; border-radius: 2px; margin: 2px 0; font-size: 11px;">'
                f'<b>{code}</b> {name}</div>',
                unsafe_allow_html=True
            )


# ============================================================================
# TAB 4: PREDICTION MAPS
# ============================================================================
with tab1:
    st.header("Prediction Maps")
    
    # Toggle to show nitrate measurements
    show_nitrate_points = st.checkbox("🧪 Show Nitrate Measurement Points", value=False, key="show_nitrate_toggle_predictions")
  
    col1, col2 = st.columns(2)
    
    with col1:
        # index_shap: continuous vulnerability index -> CONTINUOUS plotter
        norm_shap = Normalize(vmin=80, vmax=200)
        shap_map = plot_continuous_layer(data_xr, 'index_shap', cmap_vulnerability, norm_shap,
                                          title=PREDICTION_TITLES['index_shap'],
                                          lat=lat_input, lon=lon_input)
        if shap_map:
            st_folium(shap_map, width=350, height=350, key=f"shap_index_map_{lat_input}_{lon_input}")
    
    with col2:
        # index_shap_std: uncertainty (viridis) -> CONTINUOUS plotter
        norm_shap_std = Normalize(vmin=12, vmax=26)
        shap_std_map = plot_continuous_layer(data_xr, 'index_shap_std', cmap_std, norm_shap_std,
                                              title=PREDICTION_TITLES['index_shap_std'],
                                              lat=lat_input, lon=lon_input)
        if shap_std_map:
            st_folium(shap_std_map, width=350, height=350, key=f"shap_std_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # index_shap_class: 5-class vulnerability -> CLASS plotter.
        # FIX: previously routed through create_map_with_raster_overlay(), which
        # only recognized layer_name == "Risk" and otherwise always plotted
        # 'priority_zones_regulatory' - so this map was silently showing the
        # Priority layer instead of index_shap_class. Naming the variable
        # directly (as the continuous plotter already did) fixes it.
        shap_class_map = plot_class_layer(
            data_xr, 'index_shap_class',
            class_colors=vulnerability_5_colors, class_labels=vulnerability_class_labels,
            title=PREDICTION_TITLES['index_shap_class'], lat=lat_input, lon=lon_input
        )
        if shap_class_map:
            st_folium(shap_class_map, width=350, height=350, key=f"shap_class_map_{lat_input}_{lon_input}")
    
    with col2:
        # index_shap_entropy_norm: entropy (davos) -> CONTINUOUS plotter
        norm_entropy = Normalize(vmin=0, vmax=1)
        shap_entropy_map = plot_continuous_layer(data_xr, 'index_shap_entropy_norm', cmap_entropy, norm_entropy,
                                                  title=PREDICTION_TITLES['index_shap_entropy_norm'],
                                                  lat=lat_input, lon=lon_input)
        if shap_entropy_map:
            st_folium(shap_entropy_map, width=350, height=350, key=f"shap_entropy_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # y_hat: continuous NO3 concentration -> CONTINUOUS plotter
        norm_yhat = Normalize(vmin=10, vmax=100)
        y_hat_map = plot_continuous_layer(data_xr, 'y_hat', cmap_nitrate, norm_yhat,
                                           title=PREDICTION_TITLES['y_hat'],
                                           lat=lat_input, lon=lon_input)
        if y_hat_map and show_nitrate_points:
            y_hat_map = add_nitrate_layer(y_hat_map, df_nitrate_points, cmap_nitrate, norm_yhat, show_nitrate_points)
            
        if y_hat_map:
            st_folium(y_hat_map, width=350, height=350, key=f"y_hat_map_{lat_input}_{lon_input}")
    
    with col2:
        # y_hat_std: uncertainty (viridis) -> CONTINUOUS plotter
        norm_yhat_std = Normalize(vmin=5, vmax=40)
        y_hat_std_map = plot_continuous_layer(data_xr, 'y_hat_std', cmap_std, norm_yhat_std,
                                               title=PREDICTION_TITLES['y_hat_std'],
                                               lat=lat_input, lon=lon_input)
        if y_hat_std_map:
            st_folium(y_hat_std_map, width=350, height=350, key=f"y_hat_std_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # y_hat_log_class: 5-class nitrate contamination -> CLASS plotter.
        # Same fix as index_shap_class above - now reads its own variable
        # instead of silently falling back to priority_zones_regulatory.
        y_hat_class_map = plot_class_layer(
            data_xr, 'y_hat_log_class',
            class_colors=nitrate_5_colors, class_labels=nitrate_class_labels,
            title=PREDICTION_TITLES['y_hat_log_class'], lat=lat_input, lon=lon_input
        )
        if y_hat_class_map:
            st_folium(y_hat_class_map, width=350, height=350, key=f"y_hat_class_map_{lat_input}_{lon_input}")
    
    with col2:
        # y_hat_log_entropy_norm: entropy (davos) -> CONTINUOUS plotter
        norm_yhat_entropy = Normalize(vmin=0, vmax=1)
        y_hat_entropy_map = plot_continuous_layer(data_xr, 'y_hat_log_entropy_norm', cmap_entropy, norm_yhat_entropy,
                                                   title=PREDICTION_TITLES['y_hat_log_entropy_norm'],
                                                   lat=lat_input, lon=lon_input)
        if y_hat_entropy_map:
            st_folium(y_hat_entropy_map, width=350, height=350, key=f"y_hat_entropy_map_{lat_input}_{lon_input}")


# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Model
- **Method:** DRASTICLU + QRF + SHAP
- **Inputs:** 8 DRASTICLU layers
- **Outputs:** 16 variables (ranks, importance, predictions, uncertainty)
- **Classes:** Risk (1-9), Priority (1-4)
- **Uncertainty:** Quantiles + entropy
""")
