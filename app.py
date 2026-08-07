import streamlit as st
import folium
from streamlit_folium import st_folium
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
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

# SHAP Index - Continuous (blue-red)
shap_class_colors = {
    1: '#053061',
    2: '#2166AC',
    3: '#4393C3',
    4: '#F4A582',
    5: '#B2182B'
}


# Create colormaps (EXACT from notebook)
risk_ids = sorted(risk_9_colors.keys())
cmap_risk = ListedColormap([risk_9_colors[k] for k in risk_ids])
norm_risk = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap_risk.N)

priority_ids = sorted(priority_4_colors.keys())
cmap_priority = ListedColormap([priority_4_colors[k] for k in priority_ids])
norm_priority = BoundaryNorm(np.arange(0.5, 5.5, 1), cmap_priority.N)

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
# FUNCTION: Create map with raster overlay (Risk & Priority)
# ============================================================================
# ============================================================================
# FUNCTION: Create map with raster overlay (Risk & Priority - NO COLORBAR)
# ============================================================================
def create_map_with_raster_overlay(lat, lon, data_xr, layer_name, cmap_obj, norm, label_dict, color_dict):
    """Create folium map with raster data overlay + selected point (LEGEND ONLY)"""
    
    # Get data
    if layer_name == "Risk":
        raster_data = data_xr['risk_pdp_shap'].values
    else:
        raster_data = data_xr['priority_zones_regulatory'].values
    
    # Water mask
    lu_data = data_xr['LU'].values
    water_mask = (lu_data == 80)
    
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Create figure (NO padding)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='none')
    fig.patch.set_alpha(0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.patch.set_alpha(0)
    
    # Mask water
    raster_masked = np.ma.masked_where(water_mask, raster_data)
    
    # Plot (NO COLORBAR)
    im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                   cmap=cmap_obj, norm=norm, origin='lower', alpha=0.9, 
                   interpolation='nearest')
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    
    # Remove all padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    
    # Save as PNG
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, 
                facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    # Create map
    m = folium.Map(
        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Overlay (exact bounds match)
    img_url = f"data:image/png;base64,{img_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.85,
        interactive=True,
        cross_origin=False
    ).add_to(m)
    
    # Get value at point
    if layer_name == "Risk":
        layer_key = 'risk_pdp_shap'
    else:
        layer_key = 'priority_zones_regulatory'
    
    try:
        selected_value = int(float(data_xr[layer_key].sel(latitude=lat, longitude=lon, method='nearest').values))
        selected_value = max(1, min(selected_value, len(color_dict)))
    except:
        selected_value = 1
    
    # Marker color
    marker_color = color_dict.get(selected_value, 'red')
    
    # Add marker
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        popup=f"<b>{layer_name}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Value: {selected_value}",
        color=marker_color,
        fill=True,
        fillColor=marker_color,
        fillOpacity=0.95,
        weight=2
    ).add_to(m)
    
    # Legend ONLY (no colorbar)
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 160px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:9px;
                border-radius: 5px; padding: 8px; font-weight: bold;">
    {layer_name}<br>
    '''
    
    for i in sorted(color_dict.keys()):
        label = label_dict.get(i, str(i))
        legend_html += f'<div style="margin: 2px 0;"><i style="background:{color_dict[i]}; width: 12px; height: 12px; float: left; margin-right: 5px; border-radius: 1px; display: inline-block;"></i>{label}</div>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ============================================================================
# FUNCTION: Create prediction map with CORRECT RANGES & LABELS
# ============================================================================
def create_prediction_map(lat, lon, data_xr, layer_name, cmap_obj, norm_obj=None, title="", class_labels=None):
    """Create folium map for prediction layers with correct vmin/vmax"""
    
    try:
        raster_data = data_xr[layer_name].values
    except:
        return None
    
    lu_data = data_xr['LU'].values
    water_mask = (lu_data == 80)
    
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    raster_masked = np.ma.masked_where(water_mask, raster_data)
    
    # Create main map figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='none')
    fig.patch.set_alpha(0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.patch.set_alpha(0)
    
    # Plot with normalization
    if norm_obj is not None:
        im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                       cmap=cmap_obj, norm=norm_obj, origin='lower', alpha=0.9,
                       interpolation='nearest')
    else:
        im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                       cmap=cmap_obj, origin='lower', alpha=0.9,
                       interpolation='nearest')
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, 
                facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    # Create VERY SMALL horizontal colorbar
    fig_cbar, ax_cbar = plt.subplots(figsize=(1.8, 0.25), dpi=80)
    if norm_obj is not None:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj), 
                           cax=ax_cbar, orientation='horizontal', pad=0.01)
    else:
        vmin = np.nanmin(raster_masked)
        vmax = np.nanmax(raster_masked)
        norm_cont = Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cont, cmap=cmap_obj), 
                           cax=ax_cbar, orientation='horizontal', pad=0.01)
    
    cbar.ax.tick_params(labelsize=6)
    
    cbar_buffer = BytesIO()
    plt.savefig(cbar_buffer, format='png', bbox_inches='tight', dpi=80,
                facecolor='white', transparent=False, pad_inches=0.02)
    cbar_buffer.seek(0)
    cbar_base64 = base64.b64encode(cbar_buffer.read()).decode()
    plt.close()
    
    # Create folium map
    m = folium.Map(
        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Overlay
    img_url = f"data:image/png;base64,{img_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.85,
        interactive=True,
        cross_origin=False
    ).add_to(m)
    
    # Marker
    try:
        selected_value = float(data_xr[layer_name].sel(latitude=lat, longitude=lon, method='nearest').values)
        # For class labels, show the class name
        if class_labels and selected_value in class_labels:
            value_str = f"{int(selected_value)} ({class_labels[int(selected_value)]})"
        else:
            value_str = f"{selected_value:.3f}"
    except:
        selected_value = np.nan
        value_str = "N/A"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        popup=f"<b>{title}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Value: {value_str}",
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.95,
        weight=2
    ).add_to(m)
    
    # Legend with tiny horizontal colorbar and class labels
    legend_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                background-color: white; border:2px solid #333; z-index:9999; 
                border-radius: 3px; padding: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
    <div style="font-size: 10px; font-weight: bold; text-align: center; margin-bottom: 4px;">{title}</div>
    <img src="data:image/png;base64,{cbar_base64}" style="width: 160px; height: auto; display: block; margin: 0 auto 4px;">
    '''
    
    # Add class labels if provided (only for categorical maps)
    # Add class labels if provided (only for categorical maps - show names only)
    if class_labels:
        legend_html += '<div style="font-size: 8px; border-top: 1px solid #ddd; padding-top: 4px; margin-top: 2px; line-height: 1.4;">'
        for class_num in sorted(class_labels.keys()):
            legend_html += f'<div style="margin: 1px 0;">{class_labels[class_num]}</div>'
        legend_html += '</div>'
    
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
    
    # Create 4x2 grid
    for idx, config in enumerate(INPUT_LAYERS_CONFIG):
        if idx % 2 == 0:
            col1, col2 = st.columns(2)
        
        col = col1 if idx % 2 == 0 else col2
        
        with col:
            st.subheader(f"{config['title']} {config['units']}")
            
            # Get data
            try:
                layer_data = data_xr[config['layer']].values
            except:
                st.error(f"❌ Layer {config['layer']} not found")
                continue
            
            # Mask water
            lu_data = data_xr['LU'].values
            water_mask = (lu_data == 80)
            layer_masked = np.ma.masked_where(water_mask, layer_data)
            
            lats = data_xr['latitude'].values
            lons = data_xr['longitude'].values
            lat_min, lat_max = lats.min(), lats.max()
            lon_min, lon_max = lons.min(), lons.max()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='none')
            fig.patch.set_alpha(0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.patch.set_alpha(0)
            
            # Plot continuous or categorical
            if config.get('categorical'):
                # Categorical colormap - simple approach
                unique_vals = sorted(np.unique(layer_masked.compressed()))
                n_colors = len(unique_vals)
                cat_colors = [config['colors'].get(int(v), '#cccccc') for v in unique_vals]
                cat_cmap = ListedColormap(cat_colors)
                norm_cat = BoundaryNorm(np.arange(-0.5, n_colors+0.5, 1), n_colors)
                im = ax.imshow(layer_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                              cmap=cat_cmap, norm=norm_cat, origin='lower', alpha=0.9,
                              interpolation='nearest')
                
            else:
                # Continuous colormap
                vmin = config.get('vmin')
                vmax = config.get('vmax')
                
                # Calculate vmin/vmax from quantiles if not set
                if vmin is None or vmax is None:
                    try:
                        valid_data = layer_masked.compressed()
                        if len(valid_data) > 0:
                            if vmin is None and 'quantile_min' in config:
                                vmin = np.quantile(valid_data, config['quantile_min'])
                            if vmax is None and 'quantile_max' in config:
                                vmax = np.quantile(valid_data, config['quantile_max'])
                            if vmin is None:
                                vmin = valid_data.min()
                            if vmax is None:
                                vmax = valid_data.max()
                    except:
                        vmin = vmin or 0
                        vmax = vmax or 1
                
                # Ensure vmin < vmax
                if vmin is None or vmax is None or vmin >= vmax:
                    valid_data = layer_masked.compressed()
                    if len(valid_data) > 0:
                        vmin = float(np.nanmin(valid_data))
                        vmax = float(np.nanmax(valid_data))
                    else:
                        vmin = 0
                        vmax = 1
                
                if vmin == vmax:
                    vmin = vmin - 0.5
                    vmax = vmax + 0.5
                
                if config.get('log_scale'):
                    from matplotlib.colors import LogNorm
                    norm_cont = LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
                else:
                    norm_cont = Normalize(vmin=vmin, vmax=vmax)
                
                im = ax.imshow(layer_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                              cmap=config['cmap'], norm=norm_cont, origin='lower', alpha=0.9,
                              interpolation='nearest')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([lon_min, lon_max])
            ax.set_ylim([lat_min, lat_max])
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Save as PNG
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100,
                       facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            # Create horizontal colorbar for legend
            fig_cbar, ax_cbar = plt.subplots(figsize=(1.8, 0.25), dpi=80)
            if config.get('categorical'):
                cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cat, cmap=cat_cmap), 
                                   cax=ax_cbar, orientation='horizontal', pad=0.01)
            else:
                cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cont, cmap=config['cmap']), 
                                   cax=ax_cbar, orientation='horizontal', pad=0.01)
            cbar.ax.tick_params(labelsize=6)
            
            cbar_buffer = BytesIO()
            plt.savefig(cbar_buffer, format='png', bbox_inches='tight', dpi=80,
                        facecolor='white', transparent=False, pad_inches=0.02)
            cbar_buffer.seek(0)
            cbar_base64 = base64.b64encode(cbar_buffer.read()).decode()
            plt.close()
            
            # Create folium map
            m = folium.Map(
                location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom_start=10,
                tiles="OpenStreetMap"
            )
            
            # Overlay
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.85,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            
            # Add marker
            try:
                point_value = float(data_xr[config['layer']].sel(latitude=lat_input, longitude=lon_input, method='nearest').values)
                popup_text = f"{config['title']}<br>{lat_input:.4f}°N, {lon_input:.4f}°E<br>Value: {point_value:.2f}"
            except:
                popup_text = f"{config['title']}<br>{lat_input:.4f}°N, {lon_input:.4f}°E"
            
            folium.CircleMarker(
                location=[lat_input, lon_input],
                radius=6,
                popup=popup_text,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.95,
                weight=2
            ).add_to(m)
            
            # Legend with horizontal colorbar (same as prediction maps)
            legend_html = f'''
            <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                        background-color: white; border:1px solid grey; z-index:9999; 
                        border-radius: 2px; padding: 3px;">
            <div style="font-size: 9px; font-weight: bold; text-align: center; margin-bottom: 2px;">{config['title']}</div>
            <img src="data:image/png;base64,{cbar_base64}" style="width: 160px; height: auto;">
            </div>
            '''
            
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st_folium(m, width=300, height=300, key=f"input_{config['layer']}_{lat_input}_{lon_input}")
            
# ============================================================================
# TAB 1: RISK & PRIORITY MAPS
# ============================================================================
with tab3:
    st.header("Risk & Priority Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contamination Risk (DRASTICLU, 1-9)")
        risk_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Risk", cmap_risk, norm_risk, RISK_LABELS, risk_9_colors
        )
        st_folium(risk_map, width=350, height=350, key=f"risk_map_{lat_input}_{lon_input}")
    
    with col2:
        st.subheader("Management Priority (1-4)")
        priority_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Priority", cmap_priority, norm_priority, PRIORITY_LABELS, priority_4_colors
        )
        st_folium(priority_map, width=350, height=350, key=f"priority_map_{lat_input}_{lon_input}")
# ============================================================================
# TAB 2: RIVER SHAP ATTRIBUTION MAPS
# ============================================================================
with tab2:
    st.header("🎯 Driver Attribution Analysis (Rank & SHAP)")
    
    st.info("Top: Driver Rank (1-4) | Bottom: Driver SHAP values (1-4)")
    
    # Get data
    try:
        driver_rank_data = {}
        driver_shap_data = {}
        for i in range(1, 5):
            driver_rank_data[i] = data_xr[f'driver_rank_{i}'].values
            driver_shap_data[i] = data_xr[f'driver_shap_{i}'].values
    except:
        st.error("Cannot load driver data")
        st.stop()
    
    # Water mask
    lu_data = data_xr['LU'].values
    water_mask = (lu_data == 80)
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
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
    
    driver_cmap = ListedColormap([driver_colors[k] for k in range(8)])
    norm_driver = BoundaryNorm(np.arange(-0.5, 8.5, 1), 8)
    
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
            
            # Mask data
            driver_masked = np.ma.masked_where(water_mask, driver_rank_data[rank])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 7), dpi=100, facecolor='none')
            fig.patch.set_alpha(0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.patch.set_alpha(0)
            
            im = ax.imshow(driver_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                          cmap=driver_cmap, norm=norm_driver, origin='lower', alpha=0.9,
                          interpolation='nearest')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([lon_min, lon_max])
            ax.set_ylim([lat_min, lat_max])
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Save as PNG
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100,
                       facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            # Create folium map
            m = folium.Map(
                location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom_start=10,
                tiles="OpenStreetMap"
            )
            
            # Overlay
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.85,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            
            # Add marker
            try:
                driver_idx = int(data_xr[f'driver_rank_{rank}'].sel(latitude=lat_input, longitude=lon_input, method='nearest').values)
                driver_name = DRIVER_MAP.get(driver_idx, '?')
                marker_color = driver_colors.get(driver_idx, '#CCCCCC')
                popup_text = f"Rank {rank}: {driver_name}"
            except:
                marker_color = 'red'
                popup_text = f"Rank {rank}"
            
            folium.CircleMarker(
                location=[lat_input, lon_input],
                radius=5,
                popup=popup_text,
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.95,
                weight=2
            ).add_to(m)
            
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
            
            # Mask data
            shap_masked = np.ma.masked_where(water_mask, driver_shap_data[shap_rank])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 7), dpi=100, facecolor='none')
            fig.patch.set_alpha(0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.patch.set_alpha(0)
            
            im = ax.imshow(shap_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                          cmap=driver_cmap, norm=norm_driver, origin='lower', alpha=0.9,
                          interpolation='nearest')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([lon_min, lon_max])
            ax.set_ylim([lat_min, lat_max])
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Save as PNG
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100,
                       facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            # Create folium map
            m = folium.Map(
                location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom_start=10,
                tiles="OpenStreetMap"
            )
            
            # Overlay
            img_url = f"data:image/png;base64,{img_base64}"
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.85,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            
            # Add marker
            try:
                shap_idx = int(data_xr[f'driver_shap_{shap_rank}'].sel(latitude=lat_input, longitude=lon_input, method='nearest').values)
                driver_name = DRIVER_MAP.get(shap_idx, '?')
                marker_color = driver_colors.get(shap_idx, '#CCCCCC')
                popup_text = f"SHAP {shap_rank}: {driver_name}"
            except:
                marker_color = 'red'
                popup_text = f"SHAP {shap_rank}"
            
            folium.CircleMarker(
                location=[lat_input, lon_input],
                radius=5,
                popup=popup_text,
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.95,
                weight=2
            ).add_to(m)
            
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
    show_nitrate_points = st.checkbox("🧪 Show Nitrate Measurement Points", value=False, key="show_nitrate_toggle")
  
    col1, col2 = st.columns(2)
    
    with col1:
        # index_shap: 0-10 (continuous vulnerability)
        norm_shap = Normalize(vmin=80, vmax=200)
        shap_map = create_prediction_map(lat_input, lon_input, data_xr,
                                        'index_shap', cmap_vulnerability, norm_shap,
                                        title=PREDICTION_TITLES['index_shap'])
        if shap_map:
            st_folium(shap_map, width=350, height=350, key=f"shap_index_map_{lat_input}_{lon_input}")
    
    with col2:
        # index_shap_std: 12-26 (uncertainty, viridis)
        norm_shap_std = Normalize(vmin=12, vmax=26)
        shap_std_map = create_prediction_map(lat_input, lon_input, data_xr,
                                            'index_shap_std', cmap_std, norm_shap_std,
                                            title=PREDICTION_TITLES['index_shap_std'])
        if shap_std_map:
            st_folium(shap_std_map, width=350, height=350, key=f"shap_std_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # index_shap_class: 1-5 (CATEGORICAL)
        vuln_class_cmap = ListedColormap([vulnerability_5_colors[k] for k in sorted(vulnerability_5_colors.keys())])
        norm_shap_class = BoundaryNorm(np.arange(0.5, 5.5, 1), vuln_class_cmap.N)
        shap_class_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Defuzzified Specific Vulnerability", vuln_class_cmap, norm_shap_class, 
            vulnerability_class_labels, vulnerability_5_colors
        )
        st_folium(shap_class_map, width=350, height=350, key=f"shap_class_map_{lat_input}_{lon_input}")
    
    with col2:
        # index_shap_entropy_norm: 0-1 (entropy, davos)
        norm_entropy = Normalize(vmin=0, vmax=1)
        shap_entropy_map = create_prediction_map(lat_input, lon_input, data_xr,
                                                'index_shap_entropy_norm', cmap_entropy, norm_entropy,
                                                title=PREDICTION_TITLES['index_shap_entropy_norm'])
        if shap_entropy_map:
            st_folium(shap_entropy_map, width=350, height=350, key=f"shap_entropy_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # y_hat: 10-100 (continuous nitrate)
        norm_yhat = Normalize(vmin=10, vmax=100)
        y_hat_map = create_prediction_map(lat_input, lon_input, data_xr,
                                         'y_hat', cmap_nitrate, norm_yhat,
                                         title=PREDICTION_TITLES['y_hat'])
        if y_hat_map and show_nitrate_points:
            y_hat_map = add_nitrate_layer(y_hat_map, df_nitrate_points, cmap_nitrate, norm_yhat, show_nitrate_points)
            
        if y_hat_map:
            st_folium(y_hat_map, width=350, height=350, key=f"y_hat_map_{lat_input}_{lon_input}")
    
    with col2:
        # y_hat_std: 5-40 (uncertainty, viridis)
        norm_yhat_std = Normalize(vmin=5, vmax=40)
        y_hat_std_map = create_prediction_map(lat_input, lon_input, data_xr,
                                             'y_hat_std', cmap_std, norm_yhat_std,
                                             title=PREDICTION_TITLES['y_hat_std'])
        if y_hat_std_map:
            st_folium(y_hat_std_map, width=350, height=350, key=f"y_hat_std_map_{lat_input}_{lon_input}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # y_hat_log_class: 1-5 (CATEGORICAL)
        nitrate_class_cmap = ListedColormap([nitrate_5_colors[k] for k in sorted(nitrate_5_colors.keys())])
        norm_yhat_class = BoundaryNorm(np.arange(0.5, 5.5, 1), nitrate_class_cmap.N)
        y_hat_class_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Defuzzified NO₃⁻ Contamination", nitrate_class_cmap, norm_yhat_class, 
            nitrate_class_labels, nitrate_5_colors
        )
        st_folium(y_hat_class_map, width=350, height=350, key=f"y_hat_class_map_{lat_input}_{lon_input}")
    
    with col2:
        # y_hat_log_entropy_norm: 0-1 (entropy, davos)
        norm_yhat_entropy = Normalize(vmin=0, vmax=1)
        y_hat_entropy_map = create_prediction_map(lat_input, lon_input, data_xr,
                                                 'y_hat_log_entropy_norm', cmap_entropy, norm_yhat_entropy,
                                                 title=PREDICTION_TITLES['y_hat_log_entropy_norm'])
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
