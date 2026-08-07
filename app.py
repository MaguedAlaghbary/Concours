import streamlit as st
import folium
from streamlit_folium import st_folium
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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
risk_4_colors = {
    1: '#2166AC',  # Low Risk - Dark blue
    2: '#cceeff',  # Moderate Risk - Light blue
    3: '#F4A582',  # High Risk - Light red
    4: '#B2182B',  # Very High Risk - Dark red
}

# SHAP Index - Continuous (blue-red)
shap_continuous_colors = {
    'cmap': 'RdBu_r',
    'vmin': 0,
    'vmax': 10
}

# SHAP Class - Categorical
shap_class_colors = {
    1: '#053061',
    2: '#2166AC',
    3: '#4393C3',
    4: '#F4A582',
    5: '#B2182B'
}

# Y-Hat (Nitrate) - Continuous (yellow-red)
y_hat_continuous_colors = {
    'cmap': 'YlOrRd',
    'vmin': 0,
    'vmax': 400
}

# Y-Hat Class - Categorical (5 classes)
y_hat_class_colors = {
    1: '#2B8C3F',  # Very Low
    2: '#FFE135',  # Low-Moderate
    3: '#FF8D35',  # Moderate
    4: '#F72518',  # High
    5: '#67001F'   # Very High
}

# Entropy/Uncertainty - Continuous
entropy_colors = {
    'cmap': 'YlGnBu',
    'vmin': 0,
    'vmax': 1
}

# Std Dev - Continuous
std_colors = {
    'cmap': 'Greys',
    'vmin': 0,
    'vmax': 100
}

# Create colormaps (EXACT from notebook)
risk_ids = sorted(risk_9_colors.keys())
cmap_risk = ListedColormap([risk_9_colors[k] for k in risk_ids])
norm_risk = BoundaryNorm(np.arange(0.5, 10.5, 1), cmap_risk.N)

priority_ids = sorted(risk_4_colors.keys())
cmap_priority = ListedColormap([risk_4_colors[k] for k in priority_ids])
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
# SIDEBAR: LOCATION INPUT
# ============================================================================
st.sidebar.header("📍 Query Location")

col1, col2 = st.sidebar.columns(2)
with col1:
    lat_input = st.number_input("Latitude", min_value=10.9, max_value=12.7, value=11.5, step=0.01)
with col2:
    lon_input = st.number_input("Longitude", min_value=41.7, max_value=43.4, value=42.9, step=0.01)

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
def create_map_with_raster_overlay(lat, lon, data_xr, layer_name, cmap_obj, norm, label_dict):
    """Create folium map with raster data overlay + selected point"""
    
    # Get data
    if layer_name == "Risk":
        raster_data = data_xr['risk_pdp_shap'].values
        color_dict = risk_9_colors
    else:
        raster_data = data_xr['priority_zones_regulatory'].values
        color_dict = risk_4_colors
    
    # Water mask
    lu_data = data_xr['LU'].values
    water_mask = (lu_data == 80)
    
    lats = data_xr['latitude'].values
    lons = data_xr['longitude'].values
    
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80, facecolor='none')
    fig.patch.set_alpha(0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.patch.set_alpha(0)
    
    # Mask water
    raster_masked = np.ma.masked_where(water_mask, raster_data)
    
    # Plot
    im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                   cmap=cmap_obj, norm=norm, origin='lower', alpha=0.9, 
                   interpolation='nearest')
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
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
    
    # Overlay
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
    try:
        norm_value = norm(selected_value)
        rgba_color = cmap_obj(norm_value)
        import matplotlib.colors as mcolors
        marker_color = mcolors.rgb2hex(rgba_color[:3])
    except:
        marker_color = color_dict.get(selected_value, 'red')
    
    # Add marker
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        popup=f"<b>{layer_name}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Value: {selected_value}",
        color=marker_color,
        fill=True,
        fillColor=marker_color,
        fillOpacity=0.95,
        weight=2
    ).add_to(m)
    
    # Legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:11px;
                border-radius: 5px; padding: 10px">
    <b>{layer_name}</b><br>
    '''
    
    for i in sorted(color_dict.keys()):
        label = label_dict.get(i, str(i))
        legend_html += f'<i style="background:{color_dict[i]}; width: 14px; height: 14px; float: left; margin-right: 6px; border-radius: 1px;"></i><span style="font-size:10px;">{label}</span><br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ============================================================================
# FUNCTION: Create prediction map (flexible for any layer)
# ============================================================================
def create_prediction_map(lat, lon, data_xr, layer_name, cmap_config, label_dict=None):
    """Create folium map for any prediction layer"""
    
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
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80, facecolor='none')
    fig.patch.set_alpha(0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.patch.set_alpha(0)
    
    # Determine colormap type
    if isinstance(cmap_config, dict) and 'cmap' in cmap_config:
        cmap_str = cmap_config['cmap']
        vmin = cmap_config.get('vmin', raster_masked.min())
        vmax = cmap_config.get('vmax', raster_masked.max())
        im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                       cmap=cmap_str, vmin=vmin, vmax=vmax, origin='lower', alpha=0.9,
                       interpolation='nearest')
    else:
        cmap_list = ListedColormap([cmap_config[k] for k in sorted(cmap_config.keys())])
        norm_cat = BoundaryNorm(np.arange(0.5, len(cmap_config)+1.5, 1), cmap_list.N)
        im = ax.imshow(raster_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                       cmap=cmap_list, norm=norm_cat, origin='lower', alpha=0.9,
                       interpolation='nearest')
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, 
                facecolor='none', edgecolor='none', transparent=True, pad_inches=0)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    m = folium.Map(
        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    img_url = f"data:image/png;base64,{img_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.85,
        interactive=True,
        cross_origin=False
    ).add_to(m)
    
    try:
        selected_value = float(data_xr[layer_name].sel(latitude=lat, longitude=lon, method='nearest').values)
    except:
        selected_value = np.nan
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        popup=f"<b>{layer_name}</b><br>{lat:.4f}°N, {lon:.4f}°E<br>Value: {selected_value:.3f}",
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.95,
        weight=2
    ).add_to(m)
    
    legend_html = f'<div style="position: fixed; top: 10px; right: 10px; width: 200px; background-color: white; border:2px solid grey; z-index:9999; font-size:11px; border-radius: 5px; padding: 10px"><b>{layer_name}</b><br>'
    
    if label_dict:
        for i in sorted(label_dict.keys()):
            legend_html += f'{label_dict[i]}<br>'
    else:
        legend_html += f'Range: {np.nanmin(raster_masked):.2f} - {np.nanmax(raster_masked):.2f}'
    
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
    st.header("📊 Risk & Priority Maps with Full Overlay")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("(a) Contamination Risk (1-9 levels)")
        risk_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Risk", cmap_risk, norm_risk, RISK_LABELS
        )
        st_folium(risk_map, width=550, height=550, key="risk_map")
    
    with col2:
        st.subheader("(b) Management Priority (1-4 levels)")
        priority_map = create_map_with_raster_overlay(
            lat_input, lon_input, data_xr,
            "Priority", cmap_priority, norm_priority, PRIORITY_LABELS
        )
        st_folium(priority_map, width=550, height=550, key="priority_map")

# ============================================================================
# TAB 2: FEATURE IMPORTANCE
# ============================================================================
with tab2:
    st.header("🎯 Feature Importance Analysis")
    
    driver_ranks = extract_at_point(lat_input, lon_input, data_xr,
                                    [f'driver_rank_{i}' for i in range(1, 7)])
    driver_shap = extract_at_point(lat_input, lon_input, data_xr,
                                   [f'driver_shap_{i}' for i in range(1, 7)])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Driver Ranks (1=Most Important)")
        rank_data = []
        for i in range(1, 7):
            rank_val = int(driver_ranks.get(f'driver_rank_{i}', np.nan))
            if not np.isnan(rank_val):
                rank_data.append({
                    'Position': i,
                    'Feature': DRIVER_MAP.get(rank_val, f'Feature {rank_val}'),
                    'Rank': rank_val
                })
        rank_df = pd.DataFrame(rank_data)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔍 SHAP Driver Rankings")
        shap_data = []
        for i in range(1, 7):
            shap_val = driver_shap.get(f'driver_shap_{i}', np.nan)
            if not np.isnan(shap_val):
                shap_data.append({
                    'Position': i,
                    'Feature Code': int(shap_val),
                    'Feature': DRIVER_MAP.get(int(shap_val), f'Feature {int(shap_val)}')
                })
        shap_df = pd.DataFrame(shap_data)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: PREDICTIONS (Data)
# ============================================================================
with tab3:
    st.header("📈 Vulnerability Predictions")
    
    preds = extract_at_point(lat_input, lon_input, data_xr,
                            ['y_hat', 'y_hat_std', 'y_hat_log_class', 
                             'y_hat_log_entropy_norm', 'index_shap', 'index_shap_std',
                             'index_shap_class', 'index_shap_entropy_norm'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Nitrate", f"{preds.get('y_hat', np.nan):.2f} mg/L")
    
    with col2:
        st.metric("Uncertainty (±std)", f"{preds.get('y_hat_std', np.nan):.2f}")
    
    with col3:
        st.metric("SHAP Index", f"{preds.get('index_shap', np.nan):.2f}")
    
    with col4:
        st.metric("SHAP Uncertainty", f"{preds.get('index_shap_std', np.nan):.2f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        y_class = int(preds.get('y_hat_log_class', np.nan))
        st.metric("Y-Hat Class", y_class)
    
    with col2:
        entropy = preds.get('y_hat_log_entropy_norm', np.nan)
        st.metric("Y-Hat Entropy (norm)", f"{entropy:.3f}")
    
    with col3:
        shap_class = int(preds.get('index_shap_class', np.nan))
        st.metric("SHAP Class", shap_class)

# ============================================================================
# TAB 4: PREDICTION MAPS
# ============================================================================
with tab4:
    st.header("📈 Prediction Layer Maps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SHAP Index")
        shap_map = create_prediction_map(lat_input, lon_input, data_xr,
                                        'index_shap', shap_continuous_colors)
        if shap_map:
            st_folium(shap_map, width=550, height=500, key="shap_index_map")
    
    with col2:
        st.subheader("SHAP Uncertainty (Std)")
        shap_std_map = create_prediction_map(lat_input, lon_input, data_xr,
                                            'index_shap_std', std_colors)
        if shap_std_map:
            st_folium(shap_std_map, width=550, height=500, key="shap_std_map")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SHAP Class")
        shap_class_map = create_prediction_map(lat_input, lon_input, data_xr,
                                              'index_shap_class', shap_class_colors)
        if shap_class_map:
            st_folium(shap_class_map, width=550, height=500, key="shap_class_map")
    
    with col2:
        st.subheader("SHAP Entropy (Normalized)")
        shap_entropy_map = create_prediction_map(lat_input, lon_input, data_xr,
                                                'index_shap_entropy_norm', entropy_colors)
        if shap_entropy_map:
            st_folium(shap_entropy_map, width=550, height=500, key="shap_entropy_map")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Nitrate (Y-Hat)")
        y_hat_map = create_prediction_map(lat_input, lon_input, data_xr,
                                         'y_hat', y_hat_continuous_colors)
        if y_hat_map:
            st_folium(y_hat_map, width=550, height=500, key="y_hat_map")
    
    with col2:
        st.subheader("Y-Hat Uncertainty (Std)")
        y_hat_std_map = create_prediction_map(lat_input, lon_input, data_xr,
                                             'y_hat_std', std_colors)
        if y_hat_std_map:
            st_folium(y_hat_std_map, width=550, height=500, key="y_hat_std_map")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Y-Hat Class")
        y_hat_class_map = create_prediction_map(lat_input, lon_input, data_xr,
                                               'y_hat_log_class', y_hat_class_colors)
        if y_hat_class_map:
            st_folium(y_hat_class_map, width=550, height=500, key="y_hat_class_map")
    
    with col2:
        st.subheader("Y-Hat Entropy (Normalized)")
        y_hat_entropy_map = create_prediction_map(lat_input, lon_input, data_xr,
                                                 'y_hat_log_entropy_norm', entropy_colors)
        if y_hat_entropy_map:
            st_folium(y_hat_entropy_map, width=550, height=500, key="y_hat_entropy_map")

# ============================================================================
# TAB 5: COMPLETE DATA TABLE
# ============================================================================
with tab5:
    st.header("📊 Complete Data at Selected Location")
    
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
        label="📥 Download as CSV",
        data=csv,
        file_name=f"djibouti_assessment_{lat_input:.3f}_{lon_input:.3f}.csv",
        mime="text/csv"
    )

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Model Info
- **Method:** DRASTICLU + QRF + SHAP
- **24 Variables:** 8 inputs + 16 outputs
- **Risk classes:** 1-9 (Blue → Red)
- **Priority zones:** 1-4 
- **Data:** Djibouti aquifer vulnerability
- **Projection:** All data referenced to Djibouti extent
""")
