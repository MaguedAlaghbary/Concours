import streamlit as st
import folium
from streamlit_folium import st_folium
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import zipfile

st.set_page_config(page_title="Djibouti Aquifer Vulnerability", layout="wide")
st.title("🗺️ Djibouti Nitrate Vulnerability Mapper")
st.markdown("**DRASTICLU + ML-based assessment with feature importance**")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_resource
def load_data():
    # If zarr folder doesn't exist, unzip it
    if not os.path.exists('djibouti_data_minimal.zarr'):
        if os.path.exists('djibouti_data_minimal.zip'):
            with zipfile.ZipFile('djibouti_data_minimal.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
    
    data = xr.open_zarr('djibouti_data_minimal.zarr')
    return data
    return data

try:
    data_xr = load_data()
    st.success("✅ Data loaded")
except FileNotFoundError:
    st.error("❌ Missing 'djibouti_data_minimal.nc'")
    st.stop()

# ============================================================================
# DEFINE COLOR SCHEMES (from Douda notebook)
# ============================================================================

# 9-level risk colormap
risk_9_colors = {
    1: '#2B8C3F',  # Very Low - Dark Green
    2: '#6FBF4D',  # Low - Green
    3: '#B8D94D',  # Moderate-Low - Light Green
    4: '#FFE135',  # Moderate - Yellow
    5: '#FFB83C',  # Moderate-High - Orange
    6: '#FF8D35',  # High - Dark Orange
    7: '#FF5C35',  # Very High - Red-Orange
    8: '#F72518',  # Critical - Red
    9: '#8B0000'   # Emergency - Dark Red
}

# 4-level priority colormap
priority_4_colors = {
    1: '#4CAF50',  # Monitor - Green
    2: '#FFEB3B',  # Prevent - Yellow
    3: '#FFC107',  # Remediate - Amber
    4: '#F44336'   # Intervene - Red
}

# 5-level vulnerability class
vuln_5_colors = {
    1: '#2B8C3F',  # Very Low
    2: '#6FBF4D',  # Low
    3: '#FFE135',  # Moderate
    4: '#FF8D35',  # High
    5: '#F72518'   # Very High
}

# Create colormaps
risk_ids = sorted(risk_9_colors.keys())
cmap_risk = ListedColormap([risk_9_colors[k] for k in risk_ids])
norm_risk = BoundaryNorm(np.arange(0.5, 10.5, 1), cmap_risk.N)

priority_ids = sorted(priority_4_colors.keys())
cmap_priority = ListedColormap([priority_4_colors[k] for k in priority_ids])
norm_priority = BoundaryNorm(np.arange(0.5, 5.5, 1), cmap_priority.N)

vuln_ids = sorted(vuln_5_colors.keys())
cmap_vuln = ListedColormap([vuln_5_colors[k] for k in vuln_ids])
norm_vuln = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap_vuln.N)

# ============================================================================
# LABELS
# ============================================================================
RISK_LABELS = {
    1: "Very Low", 2: "Low", 3: "Mod-Low", 4: "Moderate", 5: "Mod-High",
    6: "High", 7: "Very High", 8: "Critical", 9: "Emergency"
}

PRIORITY_LABELS = {
    1: "Monitor", 2: "Prevent", 3: "Remediate", 4: "Intervene"
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
# FUNCTION: Create Folium map with legend
# ============================================================================
def create_map_with_legend(lat, lon, layer_name, cmap_dict, label_dict):
    """Create folium map with custom legend"""
    
    m = folium.Map(
        location=[11.3, 42.9],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add query point
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        popup=f"{lat:.3f}°N, {lon:.3f}°E",
        color='darkred',
        fill=True,
        fillColor='red',
        fillOpacity=0.8,
        weight=3
    ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                border-radius: 5px; padding: 10px">
    <b>{layer_name}</b><br>
    '''
    
    for i in sorted(cmap_dict.keys()):
        label = label_dict.get(i, str(i))
        legend_html += f'<i style="background:{cmap_dict[i]}; width: 18px; height: 18px; float: left; margin-right: 8px; border-radius: 2px;"></i>{label}<br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ============================================================================
# MAIN DISPLAY: TAB INTERFACE
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Risk & Priority",
    "🎯 Feature Importance", 
    "📊 Predictions",
    "📋 Data Table"
])

# ============================================================================
# TAB 1: RISK & PRIORITY MAPS
# ============================================================================
with tab1:
    st.header("Risk & Management Priority Maps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("(a) Contamination Risk (1-9 levels)")
        risk_map = create_map_with_legend(lat_input, lon_input, 
                                         "Risk Categories", 
                                         risk_9_colors, RISK_LABELS)
        st_folium(risk_map, width=500, height=500)
    
    with col2:
        st.subheader("(b) Management Priority (1-4 levels)")
        priority_map = create_map_with_legend(lat_input, lon_input,
                                             "Priority Zones",
                                             priority_4_colors, PRIORITY_LABELS)
        st_folium(priority_map, width=500, height=500)

# ============================================================================
# TAB 2: FEATURE IMPORTANCE (Driver Ranks & SHAP)
# ============================================================================
with tab2:
    st.header("🎯 Feature Importance Analysis")
    
    # Extract driver ranks and SHAP values
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
# TAB 3: PREDICTIONS & UNCERTAINTY
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
    
    # Classification
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
# TAB 4: COMPLETE DATA TABLE
# ============================================================================
with tab4:
    st.header("📊 Complete Data at Selected Location")
    
    # Extract all variables
    all_vars = list(DRASTIC_LABELS.keys()) + [
        'driver_rank_1', 'driver_rank_2', 'driver_rank_3', 'driver_rank_4', 'driver_rank_5', 'driver_rank_6',
        'driver_shap_1', 'driver_shap_2', 'driver_shap_3', 'driver_shap_4', 'driver_shap_5', 'driver_shap_6',
        'risk_pdp_shap', 'priority_zones_regulatory',
        'index_shap', 'index_shap_std', 'index_shap_class', 'index_shap_entropy_norm',
        'y_hat', 'y_hat_std', 'y_hat_log_class', 'y_hat_log_entropy_norm'
    ]
    
    data = extract_at_point(lat_input, lon_input, data_xr, all_vars)
    
    # Create table
    table_data = []
    for var, val in data.items():
        if not np.isnan(val):
            table_data.append({'Variable': var, 'Value': f"{val:.4f}"})
    
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    
    # Download option
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
- **Risk classes:** 1-9 (Very Low → Emergency)
- **Priority zones:** 1-4 (Monitor → Intervene)
- **Data:** Djibouti aquifer vulnerability
""")