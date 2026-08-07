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

tab_inputs, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 DRASTICLU Inputs",
    "🗺️ Risk & Priority",
    "🎯 Attributions", 
    "📊 Predictions",
    "📈 Prediction Maps",
    "📋 Data Table"
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
with tab1:
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
# TAB 2: FEATURE IMPORTANCE
# ============================================================================
# ============================================================================
# TAB 5: DRIVER SHAP ATTRIBUTION MAPS
# ============================================================================
with tab5:
    st.header("🎯 SHAP Driver Attribution Maps (Top 4)")
    
    st.info("Maps show which DRASTICLU parameter has highest SHAP values for ranks 1-4")
    
    # Get all 4 driver SHAP layers
    try:
        driver_shap_data = {}
        for i in range(1, 5):
            driver_shap_data[i] = data_xr[f'driver_shap_{i}'].values
    except:
        st.error("Cannot load driver SHAP data")
        st.stop()
    
    # Water mask
    lu_data = data_xr['LU'].values
    water_mask = (lu_data == 80)
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Create 2x2 grid
    cols = st.columns(2)
    
    for shap_rank in range(1, 5):
        col_idx = (shap_rank - 1) % 2
        col = cols[col_idx]
        
        with col:
            st.subheader(f"SHAP Rank {shap_rank}")
            
            # Get data
            shap_masked = np.ma.masked_where(water_mask, driver_shap_data[shap_rank])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='none')
            fig.patch.set_alpha(0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.patch.set_alpha(0)
            
            # Create colormap for drivers (0-7 = D, R, A, S, T, I, C, LU)
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
            
            # Create horizontal colorbar
            fig_cbar, ax_cbar = plt.subplots(figsize=(1.8, 0.25), dpi=80)
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_driver, cmap=driver_cmap), 
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
                shap_idx = int(data_xr[f'driver_shap_{shap_rank}'].sel(latitude=lat_input, longitude=lon_input, method='nearest').values)
                driver_name = DRIVER_MAP.get(shap_idx, '?')
                popup_text = f"SHAP Rank {shap_rank}: {driver_name}<br>{lat_input:.4f}°N, {lon_input:.4f}°E"
                marker_color = driver_colors.get(shap_idx, '#CCCCCC')
            except:
                popup_text = f"SHAP Rank {shap_rank}"
                marker_color = 'red'
            
            folium.CircleMarker(
                location=[lat_input, lon_input],
                radius=6,
                popup=popup_text,
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.95,
                weight=2
            ).add_to(m)
            
            # Legend
            legend_html = f'''
            <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                        background-color: white; border:1px solid grey; z-index:9999; 
                        border-radius: 2px; padding: 3px;">
            <div style="font-size: 9px; font-weight: bold; text-align: center; margin-bottom: 2px;">SHAP Rank {shap_rank}</div>
            <img src="data:image/png;base64,{cbar_base64}" style="width: 160px; height: auto;">
            </div>
            '''
            
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st_folium(m, width=380, height=380, key=f"driver_shap_{shap_rank}_{lat_input}_{lon_input}")
    
    # Legend panel
    st.markdown("---")
    st.subheader("📋 Parameter Color Guide (Paul Tol Bright)")
    
    legend_cols = st.columns(4)
    param_list = [
        ('D', 'Depth to Water', 1),
        ('R', 'Recharge', 2),
        ('A', 'Aquifer Media', 3),
        ('S', 'Soil Media', 4),
        ('T', 'Topography', 5),
        ('I', 'Impact Vadose', 6),
        ('C', 'Conductivity', 7),
        ('LU', 'Land Use', 8),
    ]
    
    for idx, (code, name, param_num) in enumerate(param_list):
        with legend_cols[idx % 4]:
            color = parameters_8_colors[param_num]
            st.markdown(
                f'<div style="padding: 8px; background-color: {color}; color: white; border-radius: 4px; text-align: center; font-weight: bold;">{code}<br><span style="font-size: 9px;">{name}</span></div>',
                unsafe_allow_html=True
            )
# ============================================================================
# TAB 3: PREDICTIONS (Data)
# ============================================================================
with tab3:
    st.header("Uncertainty-Quantified Predictions")
    
    preds = extract_at_point(lat_input, lon_input, data_xr,
                            ['y_hat', 'y_hat_std', 'y_hat_log_class', 
                             'y_hat_log_entropy_norm', 'index_shap', 'index_shap_std',
                             'index_shap_class', 'index_shap_entropy_norm'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Nitrate (mg/L)", f"{preds.get('y_hat', np.nan):.2f}")
    
    with col2:
        st.metric("Y-Hat Std Dev", f"{preds.get('y_hat_std', np.nan):.2f}")
    
    with col3:
        st.metric("SHAP Index", f"{preds.get('index_shap', np.nan):.2f}")
    
    with col4:
        st.metric("SHAP Std Dev", f"{preds.get('index_shap_std', np.nan):.2f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        y_class = int(preds.get('y_hat_log_class', np.nan)) if not np.isnan(preds.get('y_hat_log_class', np.nan)) else 0
        st.metric("Y-Hat Class", y_class)
    
    with col2:
        entropy = preds.get('y_hat_log_entropy_norm', np.nan)
        st.metric("Y-Hat Entropy", f"{entropy:.3f}")
    
    with col3:
        shap_class = int(preds.get('index_shap_class', np.nan)) if not np.isnan(preds.get('index_shap_class', np.nan)) else 0
        st.metric("SHAP Class", shap_class)

# ============================================================================
# TAB 4: PREDICTION MAPS
# ============================================================================
with tab4:
    st.header("Prediction Maps")
    
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
# TAB 5: COMPLETE DATA TABLE
# ============================================================================
with tab5:
    st.header("Complete Assessment Data")
    
    all_vars = list(DRASTIC_LABELS.keys()) + [
        'driver_rank_1', 'driver_rank_2', 'driver_rank_3', 'driver_rank_4', 'driver_rank_5', 'driver_rank_6',
        'driver_shap_1', 'driver_shap_2', 'driver_shap_3', 'driver_shap_4', 'driver_shap_5', 'driver_shap_6',
        'risk_pdp_shap', 'priority_zones_regulatory',
        'index_shap', 'index_shap_std', 'index_shap_class', 'index_shap_entropy_norm',
        'y_hat', 'y_hat_std', 'y_hat_log_class', 'y_hat_log_entropy_norm'
    ]
    
    data = extract_at_point(lat_input, lon_input, data_xr, all_vars)
    
    table_data = []
    for var, val in data.items():
        if not np.isnan(val):
            table_data.append({'Variable': var, 'Value': f"{val:.4f}"})
    
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    
    csv = table_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"assessment_{lat_input:.3f}_{lon_input:.3f}.csv",
        mime="text/csv"
    )

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
