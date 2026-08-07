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

# Y-Hat Class - Categorical (5 classes)
y_hat_class_colors = {
    1: '#2B8C3F',  # Very Low
    2: '#FFE135',  # Low-Moderate
    3: '#FF8D35',  # Moderate
    4: '#F72518',  # High
    5: '#67001F'   # Very High
}

# Create colormaps (EXACT from notebook)
risk_ids = sorted(risk_9_colors.keys())
cmap_risk = ListedColormap([risk_9_colors[k] for k in risk_ids])
norm_risk = BoundaryNorm(np.arange(0.5, 10.5, 1), cmap_risk.N)

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
    'davos_teal',
    ['#F0FFFF', '#A7D8DE', '#5A9FA5', '#2F5F66', '#0D2626']
)

# Use davos for entropy (from cmocean)
try:
    import cmocean.cm as cmo
    cmap_entropy = cmo.davos
except:
    # Fallback if cmocean not installed
    cmap_entropy = mcolors.LinearSegmentedColormap.from_list(
        'davos_fallback',
        ['#F0F9FF', '#B3E0F2', '#5AB4AC', '#1A7A7A', '#084B4B']
    )

# 5-class categorical for defuzzified layers (with labels)
vulnerability_class_labels = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High"
}

nitrate_class_labels = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High"
}


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Risk & Priority",
    "🎯 Feature Importance", 
    "📊 Predictions",
    "📈 Prediction Maps",
    "📋 Data Table"
])

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
with tab2:
    st.header("Feature Importance Analysis")
    
    driver_ranks = extract_at_point(lat_input, lon_input, data_xr,
                                    [f'driver_rank_{i}' for i in range(1, 7)])
    driver_shap = extract_at_point(lat_input, lon_input, data_xr,
                                   [f'driver_shap_{i}' for i in range(1, 7)])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Driver Ranks")
        rank_data = []
        for i in range(1, 7):
            rank_val = int(driver_ranks.get(f'driver_rank_{i}', np.nan))
            if not np.isnan(rank_val):
                rank_data.append({
                    'Rank': i,
                    'Feature': DRIVER_MAP.get(rank_val, f'Feature {rank_val}')
                })
        if rank_data:
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("SHAP Rankings")
        shap_data = []
        for i in range(1, 7):
            shap_val = driver_shap.get(f'driver_shap_{i}', np.nan)
            if not np.isnan(shap_val):
                shap_data.append({
                    'Rank': i,
                    'Feature': DRIVER_MAP.get(int(shap_val), f'Feature {int(shap_val)}')
                })
        if shap_data:
            st.dataframe(pd.DataFrame(shap_data), use_container_width=True, hide_index=True)

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
        norm_shap = Normalize(vmin=0, vmax=10)
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
        # index_shap_class: 1-5 (SHOW CLASS NAMES ONLY)
        norm_class = BoundaryNorm(np.arange(0.5, 6.5, 1), 256)
        shap_class_map = create_prediction_map(lat_input, lon_input, data_xr,
                                              'index_shap_class', cmap_vulnerability, norm_class,
                                              title=PREDICTION_TITLES['index_shap_class'],
                                              class_labels=vulnerability_class_labels)
        if shap_class_map:
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
        # y_hat: 10-100 (continuous nitrate CONTAMINATION, NOT concentration)
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
        # y_hat_log_class: 1-5 (SHOW CLASS NAMES ONLY)
        norm_yhat_class = BoundaryNorm(np.arange(0.5, 6.5, 1), 256)
        y_hat_class_map = create_prediction_map(lat_input, lon_input, data_xr,
                                               'y_hat_log_class', cmap_nitrate, norm_yhat_class,
                                               title=PREDICTION_TITLES['y_hat_log_class'],
                                               class_labels=nitrate_class_labels)
        if y_hat_class_map:
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
