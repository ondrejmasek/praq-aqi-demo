import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PrAQ — Prague Air Quality Forecast",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        body { font-family: Inter, system-ui, -apple-system, sans-serif; }
        .result-card-good { background: #E8F5E9; border-left: 6px solid #4CAF50; }
        .result-card-moderate { background: #FFF3E0; border-left: 6px solid #FFC107; }
        .result-card-unhealthy { background: #FFEBEE; border-left: 6px solid #F44336; }
        .result-text-good { color: #2E7D32; }
        .result-text-moderate { color: #E65100; }
        .result-text-unhealthy { color: #C62828; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & SECRETS
# ============================================================================
API_ENDPOINT = st.secrets.get("API_ENDPOINT", "https://praq-maso02-05092205.germanywestcentral.inference.ml.azure.com/score")
API_KEY = st.secrets.get("API_KEY", "")

if not API_KEY:
    st.error("⚠️ API_KEY not configured. Please add it to .streamlit/secrets.toml or environment variables.")
    st.stop()

DISTRICTS = {
    "Centre": {"east": 0, "north": 0, "south": 0, "west": 0},
    "East": {"east": 1, "north": 0, "south": 0, "west": 0},
    "North": {"east": 0, "north": 1, "south": 0, "west": 0},
    "South": {"east": 0, "north": 0, "south": 1, "west": 0},
    "West": {"east": 0, "north": 0, "south": 0, "west": 1},
}

AQI_COLORS = {
    0: {"label": "Good", "color": "#4CAF50", "hex": "good"},
    1: {"label": "Moderate", "color": "#FFC107", "hex": "moderate"},
    2: {"label": "Unhealthy", "color": "#F44336", "hex": "unhealthy"},
}

AQI_ADVICE = {
    0: "Air quality is satisfactory. Enjoy outdoor activities.",
    1: "Acceptable air quality. Sensitive individuals should limit prolonged outdoor exertion.",
    2: "Everyone may begin to experience health effects. Limit outdoor activities.",
}

# District centroids for Leaflet map (Prague, 5 zones)
DISTRICT_COORDS = {
    "Centre": [50.0755, 14.4378],
    "East": [50.1200, 14.4200],
    "North": [50.0800, 14.3200],
    "South": [50.0700, 14.5500],
    "West": [49.9900, 14.4100],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def derive_season_from_month(month):
    """Map month (1-12) to season, with autumn as reference (0 = all False)."""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:  # 9, 10, 11
        return "autumn"


def get_day_of_week(date_obj):
    """Return day_of_week: 0=Monday, 6=Sunday."""
    return date_obj.weekday()


def derive_is_weekend_from_dow(dow):
    """Return 1 if dow >= 5 (Saturday or Sunday), else 0."""
    return 1 if dow >= 5 else 0


def get_one_hot_district(district_name):
    """Return one-hot encoded district dict."""
    return DISTRICTS.get(district_name, DISTRICTS["Centre"])


def get_one_hot_season(season):
    """Return one-hot encoded season dict."""
    season_map = {
        "spring": {"spring": 1, "summer": 0, "winter": 0},
        "summer": {"spring": 0, "summer": 1, "winter": 0},
        "winter": {"spring": 0, "summer": 0, "winter": 1},
        "autumn": {"spring": 0, "summer": 0, "winter": 0},
    }
    return season_map.get(season, season_map["autumn"])


def predict_aqi(inputs):
    """
    Call Azure ML endpoint with retry logic and double JSON parsing.
    
    inputs: dict with keys:
      - temperature_mean, humidity_mean, pressure_mean, windspeed_mean, precipitation_sum
      - prev_pm25
      - month, day_of_week, is_weekend, is_holiday
      - district_name, season or (district_east/north/south/west, season_spring/summer/winter)
    
    Returns: {"prediction": 0/1/2, "label": "Good"/"Moderate"/"Unhealthy", "error": None}
             or {"prediction": None, "label": None, "error": "error message"}
    """
    
    # Build feature vector
    districts = get_one_hot_district(inputs.get("district_name", "Centre"))
    season = inputs.get("season", derive_season_from_month(inputs.get("month", 5)))
    seasons = get_one_hot_season(season)
    
    request_data = {
        "input_data": {
            "columns": [
                "temperature_mean", "humidity_mean", "pressure_mean", "windspeed_mean",
                "precipitation_sum", "prev_pm25", "month", "day_of_week", "is_weekend",
                "is_holiday", "district_east", "district_north", "district_south",
                "district_west", "season_spring", "season_summer", "season_winter"
            ],
            "index": [0],
            "data": [[
                inputs["temperature_mean"],
                inputs["humidity_mean"],
                inputs["pressure_mean"],
                inputs["windspeed_mean"],
                inputs["precipitation_sum"],
                inputs["prev_pm25"],
                inputs["month"],
                inputs["day_of_week"],
                inputs["is_weekend"],
                inputs["is_holiday"],
                districts["east"],
                districts["north"],
                districts["south"],
                districts["west"],
                seasons["spring"],
                seasons["summer"],
                seasons["winter"],
            ]]
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    max_retries = 4
    backoff_factor = 0.5
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_ENDPOINT,
                json=request_data,
                headers=headers,
                timeout=10
            )
            
            # Retry on rate limit / server errors
            if response.status_code in [429, 502, 503]:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "prediction": None,
                        "label": None,
                        "error": f"API error {response.status_code}: {response.text}"
                    }
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Handle double JSON encoding (response might be a string)
            if isinstance(data, str):
                data = json.loads(data)
            
            prediction = data.get("predictions", [None])[0]
            label = data.get("labels", [None])[0]
            
            if prediction is None or label is None:
                return {
                    "prediction": None,
                    "label": None,
                    "error": f"Invalid response format: {data}"
                }
            
            return {
                "prediction": int(prediction),
                "label": label,
                "error": None
            }
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
                continue
            return {"prediction": None, "label": None, "error": "Request timeout"}
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
                continue
            return {"prediction": None, "label": None, "error": f"Error: {str(e)}"}
    
    return {"prediction": None, "label": None, "error": "Max retries exceeded"}


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "selected_district" not in st.session_state:
    st.session_state.selected_district = "Centre"
if "result" not in st.session_state:
    st.session_state.result = None
if "all_districts_results" not in st.session_state:
    st.session_state.all_districts_results = None
if "loading" not in st.session_state:
    st.session_state.loading = False
if "override_month" not in st.session_state:
    st.session_state.override_month = None
if "override_dow" not in st.session_state:
    st.session_state.override_dow = None
if "override_is_weekend" not in st.session_state:
    st.session_state.override_is_weekend = None
if "override_season" not in st.session_state:
    st.session_state.override_season = None

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown("# PrAQ — Prague Air Quality Forecast")
st.markdown("### Predict tomorrow's air quality based on today's conditions")

col_left, col_right = st.columns([1.2, 1], gap="large")

# ============================================================================
# LEFT COLUMN — INPUT FORM
# ============================================================================

with col_left:
    st.markdown("## ☀️ Today's Conditions")
    
    # Date picker (today by default)
    today = datetime.now().date()
    selected_date = st.date_input("Select date (calculates day of week)", value=today)
    
    # Derive month and day_of_week from date
    base_month = selected_date.month
    base_dow = get_day_of_week(selected_date)
    
    # District selector with map
    st.markdown("### 📍 Select District")
    
    # Create Leaflet map
    m = folium.Map(
        location=[50.06, 14.42],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
    
    # Add district markers
    for dist_name, coords in DISTRICT_COORDS.items():
        color = "red" if dist_name == st.session_state.selected_district else "blue"
        folium.CircleMarker(
            location=coords,
            radius=8,
            popup=dist_name,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    map_data = st_folium(m, width=350, height=300)
    
    # District selector (also via radio buttons for clarity)
    district = st.radio(
        "Or select district:",
        list(DISTRICTS.keys()),
        key="selected_district",
        horizontal=True
    )
    
    st.markdown("---")
    
    # Weather sliders
    st.markdown("### 🌡️ Weather Inputs")
    
    col_temp, col_humid = st.columns(2)
    with col_temp:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-20.0, max_value=40.0, value=15.0, step=0.5
        )
    with col_humid:
        humidity = st.slider(
            "Humidity (%)",
            min_value=0, max_value=100, value=60, step=1
        )
    
    col_press, col_wind = st.columns(2)
    with col_press:
        pressure = st.slider(
            "Pressure (hPa)",
            min_value=970.0, max_value=1050.0, value=1013.0, step=0.5
        )
    with col_wind:
        windspeed = st.slider(
            "Wind speed (km/h)",
            min_value=0.0, max_value=80.0, value=5.0, step=0.5
        )
    
    col_precip, col_pm25 = st.columns(2)
    with col_precip:
        precipitation = st.slider(
            "Precipitation (mm)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.1
        )
    with col_pm25:
        prev_pm25 = st.slider(
            "Yesterday's PM2.5 (µg/m³)",
            min_value=0.0, max_value=80.0, value=12.0, step=0.5
        )
    
    st.markdown("---")
    
    # Holiday toggle
    is_holiday = st.checkbox("Is today a Czech public holiday?", value=False)
    
    st.markdown("---")
    
    # Advanced options (expandable)
    with st.expander("⚙️ Advanced Options (Edit date/season)"):
        st.markdown("**Override auto-derived values:**")
        
        col_adv_month, col_adv_dow = st.columns(2)
        with col_adv_month:
            override_month_val = st.selectbox(
                "Month (auto-derived from date)",
                options=list(range(1, 13)),
                index=base_month - 1,
                key="override_month_select"
            )
            # Sync to session state
            if override_month_val != base_month:
                st.session_state.override_month = override_month_val
            else:
                st.session_state.override_month = None
        
        with col_adv_dow:
            dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            override_dow_val = st.selectbox(
                "Day of week",
                options=list(range(7)),
                index=base_dow,
                format_func=lambda x: dow_names[x],
                key="override_dow_select"
            )
            if override_dow_val != base_dow:
                st.session_state.override_dow = override_dow_val
            else:
                st.session_state.override_dow = None
        
        col_adv_weekend, col_adv_season = st.columns(2)
        with col_adv_weekend:
            derived_is_weekend = derive_is_weekend_from_dow(base_dow)
            override_is_weekend_val = st.checkbox(
                "Is weekend",
                value=bool(derived_is_weekend),
                key="override_is_weekend_check"
            )
            if int(override_is_weekend_val) != derived_is_weekend:
                st.session_state.override_is_weekend = int(override_is_weekend_val)
            else:
                st.session_state.override_is_weekend = None
        
        with col_adv_season:
            derived_season = derive_season_from_month(base_month)
            season_options = ["Auto-derived", "Spring", "Summer", "Autumn", "Winter"]
            season_values = [None, "spring", "summer", "autumn", "winter"]
            override_season_val = st.selectbox(
                "Season",
                options=season_options,
                index=0,
                key="override_season_select"
            )
            if override_season_val != "Auto-derived":
                st.session_state.override_season = season_values[season_options.index(override_season_val)]
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict tomorrow's AQI →", use_container_width=True, type="primary"):
        st.session_state.loading = True

# ============================================================================
# RIGHT COLUMN — RESULT & ALL DISTRICTS
# ============================================================================

with col_right:
    st.markdown("## 📊 Forecast Result")
    
    if st.session_state.loading:
        with st.spinner("Fetching prediction..."):
            # Determine final values for API call
            final_month = st.session_state.override_month or selected_date.month
            final_dow = st.session_state.override_dow if st.session_state.override_dow is not None else base_dow
            final_is_weekend = st.session_state.override_is_weekend if st.session_state.override_is_weekend is not None else derive_is_weekend_from_dow(base_dow)
            final_season = st.session_state.override_season or derive_season_from_month(final_month)
            
            # Build inputs dict
            inputs = {
                "temperature_mean": temperature,
                "humidity_mean": humidity,
                "pressure_mean": pressure,
                "windspeed_mean": windspeed,
                "precipitation_sum": precipitation,
                "prev_pm25": prev_pm25,
                "month": final_month,
                "day_of_week": final_dow,
                "is_weekend": final_is_weekend,
                "is_holiday": int(is_holiday),
                "district_name": district,
                "season": final_season,
            }
            
            # Call API
            result = predict_aqi(inputs)
            st.session_state.result = result
            st.session_state.loading = False
            st.rerun()
    
    # Display result if available
    if st.session_state.result:
        result = st.session_state.result
        
        if result["error"]:
            st.error(f"❌ Prediction failed")
            with st.expander("Show error details"):
                st.code(result["error"])
        else:
            pred_idx = result["prediction"]
            pred_label = result["label"]
            pred_color = AQI_COLORS[pred_idx]["color"]
            pred_advice = AQI_ADVICE[pred_idx]
            
            # Result card
            card_class = f"result-card-{AQI_COLORS[pred_idx]['hex']}"
            text_class = f"result-text-{AQI_COLORS[pred_idx]['hex']}"
            
            st.markdown(f"""
            <div style="background: {pred_color}20; border-left: 6px solid {pred_color}; padding: 20px; border-radius: 8px; margin: 10px 0;">
                <div style="font-size: 48px; font-weight: bold; color: {pred_color}; margin-bottom: 10px;">
                    {pred_label.upper()}
                </div>
                <div style="font-size: 16px; color: #333; margin-bottom: 15px;">
                    {pred_advice}
                </div>
                <hr style="margin: 15px 0; border: none; border-top: 1px solid {pred_color}40;">
                <div style="font-size: 12px; color: #666;">
                    <strong>Today's inputs:</strong><br>
                    District: {district} | Temp: {temperature}°C | Humidity: {humidity}% | Wind: {windspeed} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # All-districts button
            if st.button("🌍 Try all districts with this weather", use_container_width=True):
                st.session_state.loading_all = True
    
    # Show all-districts results if available
    if hasattr(st.session_state, "loading_all") and st.session_state.loading_all:
        with st.spinner("Predicting for all 5 districts..."):
            final_month = st.session_state.override_month or selected_date.month
            final_dow = st.session_state.override_dow if st.session_state.override_dow is not None else base_dow
            final_is_weekend = st.session_state.override_is_weekend if st.session_state.override_is_weekend is not None else derive_is_weekend_from_dow(base_dow)
            final_season = st.session_state.override_season or derive_season_from_month(final_month)
            
            base_inputs = {
                "temperature_mean": temperature,
                "humidity_mean": humidity,
                "pressure_mean": pressure,
                "windspeed_mean": windspeed,
                "precipitation_sum": precipitation,
                "prev_pm25": prev_pm25,
                "month": final_month,
                "day_of_week": final_dow,
                "is_weekend": final_is_weekend,
                "is_holiday": int(is_holiday),
                "season": final_season,
            }
            
            all_results = {}
            for dist_name in DISTRICTS.keys():
                inputs = {**base_inputs, "district_name": dist_name}
                all_results[dist_name] = predict_aqi(inputs)
                time.sleep(1)  # Sequential with delay to avoid rate limit
            
            st.session_state.all_districts_results = all_results
            st.session_state.loading_all = False
            st.rerun()
    
    if st.session_state.all_districts_results:
        st.markdown("### 🗺️ All Districts")
        
        results = st.session_state.all_districts_results
        
        # Create a grid of 5 cards (2x2 + 1)
        col_grid1_1, col_grid1_2 = st.columns(2, gap="small")
        col_grid2_1, col_grid2_2 = st.columns(2, gap="small")
        col_grid3_1, _ = st.columns(2)
        
        dist_order = ["Centre", "East", "North", "South", "West"]
        cols = [col_grid1_1, col_grid1_2, col_grid2_1, col_grid2_2, col_grid3_1]
        
        for col, dist_name in zip(cols, dist_order):
            res = results.get(dist_name, {})
            if res.get("error"):
                with col:
                    st.warning(f"{dist_name}: Error")
            else:
                pred_idx = res.get("prediction")
                pred_color = AQI_COLORS[pred_idx]["color"]
                pred_label = AQI_COLORS[pred_idx]["label"]
                
                with col:
                    st.markdown(f"""
                    <div style="background: {pred_color}20; border-left: 4px solid {pred_color}; padding: 15px; border-radius: 6px; text-align: center;">
                        <div style="font-weight: bold; font-size: 18px; color: {pred_color}; margin-bottom: 5px;">
                            {dist_name}
                        </div>
                        <div style="font-size: 24px; font-weight: bold; color: {pred_color};">
                            {pred_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
