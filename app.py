import os
import json
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import ee
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="NDVI Prediction - Mekong Delta",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Crawled Data")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
TRAINING_DIR = os.path.join(BASE_DIR, "Training")
GEOJSON_DIR = BASE_DIR

# Constants
WIN = 16  # 16-day standard MODIS window
RAIN_MM_THRESHOLD = 1.0
WINDY_THRESH = 6.0

# Mekong Delta provinces (ĐBSCL)
MEKONG_DELTA_PROVINCES = [
    "An Giang", "Bạc Liêu", "Bến Tre", "Cà Mau", "Cần Thơ",
    "Đồng Tháp", "Hậu Giang", "Kiên Giang", "Long An",
    "Sóc Trăng", "Tiền Giang", "Trà Vinh", "Vĩnh Long"
]

# ĐBSCL province GID_1 codes (verified from actual data)
DBSCL_GID1_CODES = [
    "VNM.1_1",   # An Giang
    "VNM.2_1",   # Bạc Liêu
    "VNM.6_1",   # Bến Tre
    "VNM.13_1",  # Cà Mau
    "VNM.12_1",  # Cần Thơ
    "VNM.18_1",  # Đồng Tháp
    "VNM.24_1",  # Hậu Giang
    "VNM.33_1",  # Kiên Giang
    "VNM.39_1",  # Long An
    "VNM.51_1",  # Sóc Trăng
    "VNM.58_1",  # Tiền Giang
    "VNM.59_1",  # Trà Vinh
    "VNM.61_1"   # Vĩnh Long
]

# 54 Features for model (in exact order)
FEATURE_COLS = [
    'ndvi_mean', 'rainy_days16', 'sunny_days16', 'p95_precip1d_16', 'max_sw_dn1d_16',
    'tmax16_mean', 'tmin16_mean', 'dtr16', 'sm_diff16', 'precip_sum16',
    'precip_mean16', 'precip_std16', 'precip_p90_16', 'precip_p10_16', 'wet_frac16',
    'wet_spell_max16', 'dry_spell_max16', 'sw_dn_sum16', 'sw_dn_mean16', 'sw_dn_p90_16',
    'sw_net_sum16', 'sw_net_mean16', 'tmean16_mean', 'tmean16_std', 'tmean_p90_16',
    'tmean_p10_16', 'tmean_trend16', 'sm1_mean16', 'sm2_mean16', 'sm_diff_std16',
    'et16_sum', 'et16_mean', 'pet16_sum', 'pet16_mean', 'evap_deficit16',
    'et_pet_ratio16', 'wind_speed_mean16', 'wind_speed_max16', 'wind_speed_p90_16', 'windy_days16',
    'sp_mean16', 'sp_std16', 'year', 'date_sin', 'date_cos',
    'ndvi_mean_lag1', 'ndvi_mean_lag2', 'ndvi_mean_lag3', 'ndvi_mean_roll3_mean', 'ndvi_mean_roll3_std',
    'ndvi_mean_roll3_min', 'ndvi_mean_roll3_max', 'ndvi_mean_exp_mean', 'ndvi_mean_exp_std'
]

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data
def load_ndvi_data():
    """Load NDVI data"""
    filepath = os.path.join(DATA_DIR, "final_2016_2025_NDVI.csv")
    df = pd.read_csv(filepath)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.rename(columns={"mean": "ndvi_mean"})
    df = df.sort_values(["GID_2", "start_date"]).reset_index(drop=True)
    return df

@st.cache_data
def load_era5_data():
    """Load ERA5 climate data"""
    filepath = os.path.join(DATA_DIR, "final_2016_2025_ERA5.csv")
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["GID_2", "date"]).reset_index(drop=True)
    return df

@st.cache_data
def load_geojson_province():
    """Load province boundaries"""
    filepath = os.path.join(GEOJSON_DIR, "gadm41_VNM_1.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_geojson_district():
    """Load district boundaries"""
    filepath = os.path.join(GEOJSON_DIR, "gadm41_VNM_2.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    """Load Keras model"""
    model_path = os.path.join(MODEL_DIR, "best_fnn_model_v3.h5")
    model = keras.models.load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_scaler():
    """Create and fit scaler from training data"""
    train_filepath = os.path.join(TRAINING_DIR, "final_training_ndvi_prediction_processed.csv")
    train_df = pd.read_csv(train_filepath)
    
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)
    return scaler

@st.cache_data
def get_gid2_mapping(_ndvi_df):
    """Create GID_2 string to integer mapping"""
    unique_gids = sorted(_ndvi_df["GID_2"].unique())
    return {gid: idx for idx, gid in enumerate(unique_gids)}

@st.cache_data
def get_location_info(_ndvi_df):
    """Get unique provinces and districts (Mekong Delta only)"""
    location_df = _ndvi_df[["NAME_1", "GID_1", "NAME_2", "GID_2"]].drop_duplicates()
    # Filter to Mekong Delta provinces only
    location_df = location_df[location_df["NAME_1"].isin(MEKONG_DELTA_PROVINCES)]
    provinces = sorted(location_df["NAME_1"].unique())
    return location_df, provinces

# ============================================================================
# GOOGLE EARTH ENGINE FUNCTIONS
# ============================================================================
@st.cache_resource
def initialize_gee():
    """Initialize Google Earth Engine with service account"""
    try:
        key_path = os.path.join(BASE_DIR, "key", "peciatas-65446e8be41e.json") # <-- Path to your service account key file
        credentials = ee.ServiceAccountCredentials(None, key_path)
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"GEE initialization failed: {e}")
        return False

def get_latest_16day_window():
    """Find the latest complete 16-day window from available ERA5 data
    
    Strategy:
    1. Find the latest date in ERA5 dataset
    2. Work backwards to find a complete 16-day window
    3. This ensures we have sufficient data for prediction
    
    Returns:
        input_start: Start of input window
        input_end: End of input window (last day of ERA5 data)
        predict_start: Prediction date (input_end + 1 day)
        predict_end: End of predicted period (predict_start + 15 days)
    """
    # Load ERA5 data to find latest available date
    era5_df = load_era5_data()
    
    # Filter to ĐBSCL only using GID_1 codes
    dbscl_era5 = era5_df[era5_df["GID_1"].isin(DBSCL_GID1_CODES)]
    
    if len(dbscl_era5) == 0:
        # Fallback to today if no data
        today = dt.datetime.now().date()
        input_end = today - timedelta(days=1)
        input_start = input_end - timedelta(days=15)
        predict_start = today
        predict_end = today + timedelta(days=15)
        return input_start, input_end, predict_start, predict_end
    
    # Get latest date with good coverage (at least 50% of districts have data)
    latest_date = dbscl_era5["date"].max()
    
    # Work backwards to ensure 16-day window
    input_end = latest_date
    input_start = input_end - timedelta(days=15)  # 16 days total
    predict_start = input_end + timedelta(days=1)
    predict_end = predict_start + timedelta(days=15)
    
    return input_start, input_end, predict_start, predict_end

@st.cache_data
def fetch_gee_ndvi_for_district(gid2: str, start_date: dt.date, end_date: dt.date, _geojson_district):
    """Fetch NDVI data from GEE for a specific district"""
    try:
        # Find district geometry
        district_feature = None
        for feat in _geojson_district["features"]:
            if feat["properties"]["GID_2"] == gid2:
                district_feature = feat
                break
        
        if not district_feature:
            return None
        
        # Convert to EE geometry
        geometry = ee.Geometry(district_feature["geometry"])
        
        # Load MODIS NDVI (MOD13Q1 - 250m 16-day)
        ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(start_date.strftime("%Y-%m-%d"), (end_date + timedelta(days=1)).strftime("%Y-%m-%d")) \
            .filterBounds(geometry) \
            .select("NDVI")
        
        # Get mean NDVI
        if ndvi_collection.size().getInfo() == 0:
            return None
        
        ndvi_image = ndvi_collection.mean()
        stats = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=250,
            maxPixels=1e9
        ).getInfo()
        
        ndvi_value = stats.get("NDVI")
        if ndvi_value is not None:
            # MODIS NDVI is scaled by 10000
            return float(ndvi_value) / 10000.0
        return None
        
    except Exception as e:
        st.warning(f"GEE NDVI fetch error for {gid2}: {e}")
        return None

@st.cache_data
def fetch_gee_era5_for_district(gid2: str, start_date: dt.date, end_date: dt.date, _geojson_district):
    """Fetch ERA5-Land data from GEE for a specific district"""
    try:
        # Find district geometry
        district_feature = None
        for feat in _geojson_district["features"]:
            if feat["properties"]["GID_2"] == gid2:
                district_feature = feat
                break
        
        if not district_feature:
            return None
        
        geometry = ee.Geometry(district_feature["geometry"])
        
        # ERA5-Land daily aggregates
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date.strftime("%Y-%m-%d"), (end_date + timedelta(days=1)).strftime("%Y-%m-%d")) \
            .filterBounds(geometry)
        
        if era5.size().getInfo() == 0:
            return None
        
        # Get all bands needed
        bands = [
            "temperature_2m",
            "total_precipitation_sum", 
            "surface_solar_radiation_downwards_sum",
            "surface_net_solar_radiation_sum",
            "total_evaporation_sum",
            "potential_evaporation_sum",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
            "surface_pressure"
        ]
        
        # Aggregate to dataframe
        data_list = []
        collection_list = era5.toList(era5.size())
        
        for i in range(era5.size().getInfo()):
            image = ee.Image(collection_list.get(i))
            date_str = image.date().format("YYYY-MM-dd").getInfo()
            
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=11132,  # ERA5-Land resolution ~11km
                maxPixels=1e9
            ).getInfo()
            
            row = {"date": date_str}
            for band in bands:
                value = stats.get(band)
                row[band] = float(value) if value is not None else np.nan
            
            data_list.append(row)
        
        if not data_list:
            return None
            
        df = pd.DataFrame(data_list)
        df["date"] = pd.to_datetime(df["date"])
        
        # Rename columns to match training data
        df = df.rename(columns={
            "temperature_2m": "2m_temperature_mean",
            "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1_mean",
            "volumetric_soil_water_layer_2": "volumetric_soil_water_layer_2_mean",
            "u_component_of_wind_10m": "10m_u_component_of_wind_mean",
            "v_component_of_wind_10m": "10m_v_component_of_wind_mean",
            "surface_pressure": "surface_pressure_mean"
        })
        
        # Convert temperature from K to C
        if "2m_temperature_mean" in df.columns:
            df["2m_temperature_mean"] = df["2m_temperature_mean"] - 273.15
        
        # Convert precipitation from m to mm
        if "total_precipitation_sum" in df.columns:
            df["total_precipitation_sum"] = df["total_precipitation_sum"] * 1000
        
        # Convert pressure from Pa to hPa
        if "surface_pressure_mean" in df.columns:
            df["surface_pressure_mean"] = df["surface_pressure_mean"] / 100
        
        return df
        
    except Exception as e:
        st.warning(f"GEE ERA5 fetch error for {gid2}: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS (from original app)
# ============================================================================
def run_lengths(mask_bool: np.ndarray):
    """Calculate lengths of consecutive True runs"""
    if mask_bool.size == 0:
        return []
    m = mask_bool.astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return (ends - starts).tolist()

def nan_slope(y):
    """Compute linear regression slope ignoring NaN"""
    y = np.asarray(y, dtype=float)
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return np.nan
    x_val = np.arange(len(y))[valid_mask]
    y_val = y[valid_mask]
    if len(x_val) < 2:
        return np.nan
    slope, _ = np.polyfit(x_val, y_val, 1)
    return float(slope)

def to_mm_positive(arr_like):
    """Convert ET/PET to positive mm"""
    x = np.asarray(arr_like, dtype=float)
    if np.nanmedian(x) < 0:
        x = -x
    return x

def compute_window_features(era5_window: pd.DataFrame, solar_p75: float):
    """Compute 42 window-based features from 16-day ERA5 data"""
    # Basic vectors
    precip = era5_window["total_precipitation_sum"].to_numpy()
    tmean = era5_window["2m_temperature_mean"].to_numpy()
    sm1 = era5_window["volumetric_soil_water_layer_1_mean"].to_numpy()
    sm2 = era5_window["volumetric_soil_water_layer_2_mean"].to_numpy()
    sw_dn = era5_window["surface_solar_radiation_downwards_sum"].to_numpy()
    
    sunny = sw_dn > solar_p75
    wet = precip > RAIN_MM_THRESHOLD
    
    # Main features
    rainy_days16 = int(np.nansum(wet))
    sunny_days16 = int(np.nansum(sunny))
    p95_precip1d_16 = float(np.nanpercentile(precip, 95))
    max_sw_dn1d_16 = float(np.nanmax(sw_dn))
    sm_diff16 = float(np.nanmean(sm1 - sm2))
    
    # Temperature proxy
    tmax16_mean = float(np.nanpercentile(tmean, 90))
    tmin16_mean = float(np.nanpercentile(tmean, 10))
    dtr16 = float(tmax16_mean - tmin16_mean)
    
    # Precip summary
    precip_sum16 = float(np.nansum(precip))
    precip_mean16 = float(np.nanmean(precip))
    precip_std16 = float(np.nanstd(precip, ddof=0))
    precip_p90_16 = float(np.nanpercentile(precip, 90))
    precip_p10_16 = float(np.nanpercentile(precip, 10))
    wet_frac16 = float(rainy_days16 / len(era5_window)) if len(era5_window) > 0 else np.nan
    
    wet_runs = run_lengths(wet)
    dry_runs = run_lengths(~wet)
    wet_spell_max16 = int(max(wet_runs) if wet_runs else 0)
    dry_spell_max16 = int(max(dry_runs) if dry_runs else 0)
    
    # Solar summary
    sw_dn_sum16 = float(np.nansum(sw_dn))
    sw_dn_mean16 = float(np.nanmean(sw_dn))
    sw_dn_p90_16 = float(np.nanpercentile(sw_dn, 90))
    
    # Net solar
    if "surface_net_solar_radiation_sum" in era5_window.columns:
        sw_net = era5_window["surface_net_solar_radiation_sum"].to_numpy()
        sw_net_sum16 = float(np.nansum(sw_net))
        sw_net_mean16 = float(np.nanmean(sw_net))
    else:
        sw_net_sum16 = np.nan
        sw_net_mean16 = np.nan
    
    # Temperature summary
    tmean16_mean = float(np.nanmean(tmean))
    tmean16_std = float(np.nanstd(tmean, ddof=0))
    tmean_p90_16 = float(np.nanpercentile(tmean, 90))
    tmean_p10_16 = float(np.nanpercentile(tmean, 10))
    tmean_trend16 = float(nan_slope(tmean))
    
    # Soil moisture
    sm1_mean16 = float(np.nanmean(sm1))
    sm2_mean16 = float(np.nanmean(sm2))
    sm_diff_std16 = float(np.nanstd(sm1 - sm2, ddof=0))
    
    # Evapotranspiration
    if "total_evaporation_sum" in era5_window.columns:
        et_mm = to_mm_positive(era5_window["total_evaporation_sum"].to_numpy())
        et16_sum = float(np.nansum(et_mm))
        et16_mean = float(np.nanmean(et_mm))
    else:
        et16_sum = np.nan
        et16_mean = np.nan
        et_mm = None
    
    if "potential_evaporation_sum" in era5_window.columns:
        pet_mm = to_mm_positive(era5_window["potential_evaporation_sum"].to_numpy())
        pet16_sum = float(np.nansum(pet_mm))
        pet16_mean = float(np.nanmean(pet_mm))
    else:
        pet16_sum = np.nan
        pet16_mean = np.nan
        pet_mm = None
    
    if et_mm is not None and pet_mm is not None:
        evap_deficit16 = float(np.nansum(np.maximum(pet_mm - et_mm, 0.0)))
        et_pet_ratio16 = float(et16_sum / pet16_sum) if pet16_sum and pet16_sum != 0 else np.nan
    else:
        evap_deficit16 = np.nan
        et_pet_ratio16 = np.nan
    
    # Wind speed
    if {"10m_u_component_of_wind_mean", "10m_v_component_of_wind_mean"}.issubset(era5_window.columns):
        u = era5_window["10m_u_component_of_wind_mean"].to_numpy()
        v = era5_window["10m_v_component_of_wind_mean"].to_numpy()
        wind_speed = np.hypot(u, v)
        wind_speed_mean16 = float(np.nanmean(wind_speed))
        wind_speed_max16 = float(np.nanmax(wind_speed))
        wind_speed_p90_16 = float(np.nanpercentile(wind_speed, 90))
        windy_days16 = int(np.nansum(wind_speed > WINDY_THRESH))
    else:
        wind_speed_mean16 = np.nan
        wind_speed_max16 = np.nan
        wind_speed_p90_16 = np.nan
        windy_days16 = 0
    
    # Surface pressure
    if "surface_pressure_mean" in era5_window.columns:
        sp = era5_window["surface_pressure_mean"].to_numpy()
        sp_mean16 = float(np.nanmean(sp))
        sp_std16 = float(np.nanstd(sp, ddof=0))
    else:
        sp_mean16 = np.nan
        sp_std16 = np.nan
    
    return {
        "rainy_days16": rainy_days16,
        "sunny_days16": sunny_days16,
        "p95_precip1d_16": p95_precip1d_16,
        "max_sw_dn1d_16": max_sw_dn1d_16,
        "tmax16_mean": tmax16_mean,
        "tmin16_mean": tmin16_mean,
        "dtr16": dtr16,
        "sm_diff16": sm_diff16,
        "precip_sum16": precip_sum16,
        "precip_mean16": precip_mean16,
        "precip_std16": precip_std16,
        "precip_p90_16": precip_p90_16,
        "precip_p10_16": precip_p10_16,
        "wet_frac16": wet_frac16,
        "wet_spell_max16": wet_spell_max16,
        "dry_spell_max16": dry_spell_max16,
        "sw_dn_sum16": sw_dn_sum16,
        "sw_dn_mean16": sw_dn_mean16,
        "sw_dn_p90_16": sw_dn_p90_16,
        "sw_net_sum16": sw_net_sum16,
        "sw_net_mean16": sw_net_mean16,
        "tmean16_mean": tmean16_mean,
        "tmean16_std": tmean16_std,
        "tmean_p90_16": tmean_p90_16,
        "tmean_p10_16": tmean_p10_16,
        "tmean_trend16": tmean_trend16,
        "sm1_mean16": sm1_mean16,
        "sm2_mean16": sm2_mean16,
        "sm_diff_std16": sm_diff_std16,
        "et16_sum": et16_sum,
        "et16_mean": et16_mean,
        "pet16_sum": pet16_sum,
        "pet16_mean": pet16_mean,
        "evap_deficit16": evap_deficit16,
        "et_pet_ratio16": et_pet_ratio16,
        "wind_speed_mean16": wind_speed_mean16,
        "wind_speed_max16": wind_speed_max16,
        "wind_speed_p90_16": wind_speed_p90_16,
        "windy_days16": windy_days16,
        "sp_mean16": sp_mean16,
        "sp_std16": sp_std16,
    }

def compute_lag_features(ndvi_history: pd.DataFrame):
    """Compute 9 NDVI lag features"""
    ndvi_values = ndvi_history["ndvi_mean"].to_numpy()
    
    # Lag features (t-1, t-2, t-3)
    ndvi_mean_lag1 = float(ndvi_values[-1]) if len(ndvi_values) >= 1 else np.nan
    ndvi_mean_lag2 = float(ndvi_values[-2]) if len(ndvi_values) >= 2 else np.nan
    ndvi_mean_lag3 = float(ndvi_values[-3]) if len(ndvi_values) >= 3 else np.nan
    
    # Rolling window features (last 3 periods)
    if len(ndvi_values) >= 3:
        last_3 = ndvi_values[-3:]
        ndvi_mean_roll3_mean = float(np.mean(last_3))
        ndvi_mean_roll3_std = float(np.std(last_3, ddof=0))
        ndvi_mean_roll3_min = float(np.min(last_3))
        ndvi_mean_roll3_max = float(np.max(last_3))
    else:
        ndvi_mean_roll3_mean = np.nan
        ndvi_mean_roll3_std = np.nan
        ndvi_mean_roll3_min = np.nan
        ndvi_mean_roll3_max = np.nan
    
    # Expanding window features (all history)
    if len(ndvi_values) > 0:
        ndvi_mean_exp_mean = float(np.mean(ndvi_values))
        ndvi_mean_exp_std = float(np.std(ndvi_values, ddof=0))
    else:
        ndvi_mean_exp_mean = np.nan
        ndvi_mean_exp_std = np.nan
    
    return {
        "ndvi_mean_lag1": ndvi_mean_lag1,
        "ndvi_mean_lag2": ndvi_mean_lag2,
        "ndvi_mean_lag3": ndvi_mean_lag3,
        "ndvi_mean_roll3_mean": ndvi_mean_roll3_mean,
        "ndvi_mean_roll3_std": ndvi_mean_roll3_std,
        "ndvi_mean_roll3_min": ndvi_mean_roll3_min,
        "ndvi_mean_roll3_max": ndvi_mean_roll3_max,
        "ndvi_mean_exp_mean": ndvi_mean_exp_mean,
        "ndvi_mean_exp_std": ndvi_mean_exp_std,
    }

def compute_time_features(pred_date: dt.date):
    """Compute 3 time features"""
    year = pred_date.year
    day_of_year = pred_date.timetuple().tm_yday
    
    # Normalize by days in year
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    day_norm = day_of_year / days_in_year
    
    # Cyclical encoding
    date_sin = float(np.sin(2 * np.pi * day_norm))
    date_cos = float(np.cos(2 * np.pi * day_norm))
    
    return {
        "year": year,
        "date_sin": date_sin,
        "date_cos": date_cos,
    }

def prepare_features(gid2, pred_date, era5_df, ndvi_df, gid2_mapping, scaler):
    """Prepare all 54 features + GID_2 embedding for prediction"""
    
    # === 1. ERA5 Window Features (42 features) ===
    window_start = pred_date - timedelta(days=WIN)
    window_end = pred_date - timedelta(days=1)
    
    era5_window = era5_df[
        (era5_df["GID_2"] == gid2) & 
        (era5_df["date"] >= window_start) & 
        (era5_df["date"] <= window_end)
    ]
    
    # Allow predictions with minimum 15 days (94% of standard 16 days)
    if len(era5_window) < 15:
        return None, None, None, None, f"Insufficient ERA5 data: need 15+ days, got {len(era5_window)}"
    
    # Warning if data is incomplete but usable
    data_warning = None
    if len(era5_window) < WIN:
        data_warning = f"Incomplete ERA5 data: {len(era5_window)}/{WIN} days"
    
    # Compute p75 solar radiation for this district
    era5_district = era5_df[era5_df["GID_2"] == gid2]
    solar_p75 = era5_district["surface_solar_radiation_downwards_sum"].quantile(0.75)
    
    window_feats = compute_window_features(era5_window, solar_p75)
    
    # === 2. Time Features (3 features) ===
    time_feats = compute_time_features(pred_date)
    
    # === 3. NDVI Lag Features (9 features) ===
    ndvi_history = ndvi_df[
        (ndvi_df["GID_2"] == gid2) & 
        (ndvi_df["end_date"] < pred_date)
    ].sort_values("start_date")
    
    if len(ndvi_history) < 3:
        return None, None, None, None, "Insufficient NDVI history: need at least 3 periods"
    
    lag_feats = compute_lag_features(ndvi_history)
    
    # === 4. Current NDVI (1 feature) ===
    current_ndvi_row = ndvi_history.iloc[-1]
    current_ndvi = float(current_ndvi_row["ndvi_mean"])
    
    # === Combine all features ===
    all_features = {"ndvi_mean": current_ndvi}
    all_features.update(window_feats)
    all_features.update(time_feats)
    all_features.update(lag_feats)
    
    # Create feature array in correct order
    feature_array = np.array([[all_features[col] for col in FEATURE_COLS]], dtype=np.float32)
    
    # Scale features
    feature_array_scaled = scaler.transform(feature_array)
    
    # GID_2 embedding
    gid2_encoded = np.array([[gid2_mapping[gid2]]], dtype=np.int32)
    
    return feature_array_scaled, gid2_encoded, era5_window, all_features, data_warning

# ============================================================================
# MAP FUNCTIONS
# ============================================================================
def get_ndvi_color(ndvi_value):
    """Map NDVI value to color"""
    if ndvi_value < 0.2:
        return "#8B4513"  # Bare soil
    elif ndvi_value < 0.4:
        return "#FFD700"  # Sparse
    elif ndvi_value < 0.6:
        return "#90EE90"  # Moderate
    elif ndvi_value < 0.8:
        return "#228B22"  # Dense
    else:
        return "#006400"  # Very dense

def get_ndvi_category(ndvi_value):
    """Get NDVI category label"""
    if ndvi_value < 0.2:
        return "Bare Soil / Water"
    elif ndvi_value < 0.4:
        return "Sparse Vegetation"
    elif ndvi_value < 0.6:
        return "Moderate Vegetation"
    elif ndvi_value < 0.8:
        return "Dense Vegetation"
    else:
        return "Very Dense Vegetation"

def create_map(geojson_province, geojson_district, selected_gid1, ndvi_dict=None, title="Map", highlighted_gid2=None, location_df=None):
    """Create Folium map with NDVI heatmap coloring
    
    Args:
        geojson_province: Province GeoJSON
        geojson_district: District GeoJSON
        selected_gid1: Selected province GID_1 (or None for all provinces)
        ndvi_dict: Dictionary mapping GID_2 -> NDVI value for coloring
        title: Map title
        highlighted_gid2: GID_2 of district to highlight with thick border
        location_df: DataFrame with proper district names (NAME_1, NAME_2, GID_2)
    """
    
    # Default to Mekong Delta region
    default_center = (9.5, 105.8)
    default_zoom = 8
    
    # Determine map center and zoom
    if selected_gid1:
        province_feature = None
        for feature in geojson_province["features"]:
            if feature["properties"].get("GID_1") == selected_gid1:
                province_feature = feature
                break
        
        if province_feature:
            coords = []
            geom = province_feature["geometry"]
            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
            elif geom["type"] == "MultiPolygon":
                for poly in geom["coordinates"]:
                    coords.extend(poly[0])
            
            if coords:
                lats = [c[1] for c in coords]
                lons = [c[0] for c in coords]
                center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
                zoom = 9
            else:
                center, zoom = default_center, default_zoom
        else:
            center, zoom = default_center, default_zoom
    else:
        center, zoom = default_center, default_zoom
    
    # Create map with zoom/drag disabled for stability
    m = folium.Map(
        location=center, 
        zoom_start=zoom, 
        tiles=None,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False,
        doubleClickZoom=False
    )
    
    # Add tile layer
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='&copy; OpenStreetMap contributors &copy; CARTO',
        name='CartoDB Light (No Labels)',
        control=False
    ).add_to(m)
    
    # If no province selected, show only ĐBSCL provinces with prominent borders
    if not selected_gid1:
        for feature in geojson_province["features"]:
            gid1 = feature["properties"].get("GID_1")
            
            # Only show ĐBSCL provinces using GID_1 codes
            if gid1 in DBSCL_GID1_CODES:
                # Get proper name from location_df
                if location_df is not None:
                    loc_row = location_df[location_df["GID_1"] == gid1]
                    if not loc_row.empty:
                        name1 = loc_row.iloc[0]["NAME_1"]
                    else:
                        name1 = feature["properties"].get("NAME_1", "Unknown")
                else:
                    name1 = feature["properties"].get("NAME_1", "Unknown")
                
                # ĐBSCL province styling with prominent black borders
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {
                        "fillColor": "#f5f5f5",  # Light gray fill
                        "color": "#000000",      # Black border
                        "weight": 3,             # Thicker border for prominence
                        "fillOpacity": 0.4,      # Slightly more visible fill
                    },
                    tooltip=f"<b>{name1}</b><br><i>Click to view details</i>"
                ).add_to(m)
        return m
    
    # Province selected - show districts with NDVI coloring
    for feature in geojson_district["features"]:
        if feature["properties"].get("GID_1") == selected_gid1:
            gid2 = feature["properties"].get("GID_2")
            
            # Get proper names from location_df if available
            if location_df is not None:
                loc_row = location_df[location_df["GID_2"] == gid2]
                if not loc_row.empty:
                    name1 = loc_row.iloc[0]["NAME_1"]
                    name2 = loc_row.iloc[0]["NAME_2"]
                else:
                    name1 = feature["properties"].get("NAME_1", "Unknown")
                    name2 = feature["properties"].get("NAME_2", "Unknown")
            else:
                name1 = feature["properties"].get("NAME_1", "Unknown")
                name2 = feature["properties"].get("NAME_2", "Unknown")
            
            # Get NDVI value for this district if available
            if ndvi_dict and gid2 in ndvi_dict:
                ndvi_val = ndvi_dict[gid2]
                fill_color = get_ndvi_color(ndvi_val)
                fill_opacity = 0.7
                tooltip_text = f"<b>Province:</b> {name1}<br><b>District:</b> {name2}<br><b>NDVI:</b> {ndvi_val:.4f}"
            else:
                # No NDVI data - show gray with lighter opacity
                fill_color = "#d3d3d3"
                fill_opacity = 0.3
                tooltip_text = f"<b>Province:</b> {name1}<br><b>District:</b> {name2}<br><i>Missing data</i>"
            
            # Highlight selected district with thick black border (remove old border style)
            is_highlighted = (highlighted_gid2 and gid2 == highlighted_gid2)
            if is_highlighted:
                # Selected district: thick black border, no gray border
                border_color = "#000000"
                border_weight = 4
            else:
                # Unselected districts: thin gray border
                border_color = "#666666"
                border_weight = 1
            
            folium.GeoJson(
                feature,
                style_function=lambda x, fc=fill_color, fo=fill_opacity, bc=border_color, bw=border_weight: {
                    "fillColor": fc,
                    "color": bc,
                    "weight": bw,
                    "fillOpacity": fo,
                },
                tooltip=tooltip_text
            ).add_to(m)
    
    return m

def create_ndvi_legend():
    """Create NDVI legend with max-height matching maps"""
    st.markdown(
        "<div style='margin-top:78px; margin-bottom:6px; font-size:1.25rem; font-weight:600; color:var(--text-color);'>NDVI Scale</div>",
        unsafe_allow_html=True
    )
    
    legend_data = [
        ("0.0-0.2", "Bare Soil", "#8B4513"),
        ("0.2-0.4", "Sparse", "#FFD700"),
        ("0.4-0.6", "Moderate", "#90EE90"),
        ("0.6-0.8", "Dense", "#228B22"),
        ("0.8-1.0", "Very Dense", "#006400"),
    ]
    
    for range_str, description, color in legend_data:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:start;margin-bottom:8px;">
                <div style="width:24px;height:16px;background-color:{color};margin-right:8px;border:1px solid #ddd;"></div>
                <span style="font-size:12px;color:var(--text-color);">{range_str} · {description}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Minimalist header
    st.title("NDVI Prediction - Mekong Delta")
    st.markdown("<p style='color:#fff;font-size:14px;'>Automated NDVI prediction using satellite data | Select a province to view predictions</p>", unsafe_allow_html=True)
    st.divider()
    
    # Initialize GEE
    gee_initialized = initialize_gee()
    if not gee_initialized:
        st.error("❌ Failed to initialize Google Earth Engine. Check service account key.")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        ndvi_df = load_ndvi_data()
        era5_df = load_era5_data()
        geojson_province = load_geojson_province()
        geojson_district = load_geojson_district()
        model = load_model()
        scaler = load_scaler()
        gid2_mapping = get_gid2_mapping(ndvi_df)
        location_df, provinces = get_location_info(ndvi_df)
    
    # Sidebar
    st.sidebar.header("Selection")
    
    # Province selection
    province_options = [""] + list(provinces)
    st.sidebar.markdown("""
        <style>
        div[data-testid="stSelectbox"] > div:first-child {
            background-color: rgba(0, 255, 0, 0.1);
            border-radius: 5px;
            padding: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    selected_province = st.sidebar.selectbox(
        "Select Province",
        options=province_options,
        index=0,
        format_func=lambda x: "-- Select Province --" if x == "" else x
    )
    
    # Get GID_1 for selected province
    gid1 = None
    if selected_province:
        prov_row = location_df[location_df["NAME_1"] == selected_province].iloc[0]
        gid1 = prov_row["GID_1"]
    
    # Get prediction window based on TODAY
    input_start, input_end, predict_start, predict_end = get_latest_16day_window()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **📅 Prediction Window**
    
    **Input Data:**  
    {input_start.strftime('%Y-%m-%d')} → {input_end.strftime('%Y-%m-%d')}  
    _(16 days of weather data)_
    
    **Predicted NDVI Period:**  
    {predict_start.strftime('%Y-%m-%d')} → {predict_end.strftime('%Y-%m-%d')}  
    _(16-day composite)_
    
    **Data Status:**  
    Latest available ERA5 data used
    """)
    
    # Auto-prediction when province selected
    current_ndvi_dict = {}
    predicted_ndvi_dict = {}
    prediction_details = {}  # Store detailed info for each district
    
    if selected_province and gid1:
        st.sidebar.markdown("---")
        st.sidebar.success(f"**Selected:** {selected_province}")
        
        # Get all districts in province
        districts_in_province = location_df[location_df["NAME_1"] == selected_province]
        
        with st.spinner(f"Predicting for {len(districts_in_province)} districts using historical data..."):
            progress_bar = st.progress(0)
            for idx, (_, row) in enumerate(districts_in_province.iterrows()):
                gid2 = row["GID_2"]
                district_name = row["NAME_2"]
                
                try:
                    # Get latest NDVI from historical data (for current NDVI display)
                    district_ndvi = ndvi_df[ndvi_df["GID_2"] == gid2].sort_values("start_date")
                    if len(district_ndvi) > 0:
                        latest_ndvi = float(district_ndvi.iloc[-1]["ndvi_mean"])
                        current_ndvi_dict[gid2] = latest_ndvi
                    
                    # Prepare features and predict
                    if len(district_ndvi) >= 3:  # Need at least 3 NDVI periods for lag features
                        predict_date = predict_start
                        
                        feature_array, gid2_encoded, era5_window, all_features, data_warning = prepare_features(
                            gid2, predict_date, era5_df, ndvi_df, gid2_mapping, scaler
                        )
                        
                        if feature_array is not None and data_warning is None or (data_warning and data_warning.startswith("Incomplete")):
                            # Predict even with incomplete data (will have warning)
                            y_pred = model([feature_array, gid2_encoded], training=False)
                            predicted_ndvi = float(y_pred.numpy()[0][0])
                            predicted_ndvi = np.clip(predicted_ndvi, 0.0, 1.0)
                            predicted_ndvi_dict[gid2] = predicted_ndvi
                            
                            # Store details for this district
                            prediction_details[gid2] = {
                                "name": district_name,
                                "current_ndvi": latest_ndvi,
                                "predicted_ndvi": predicted_ndvi,
                                "era5_window": era5_window,
                                "all_features": all_features,
                                "predict_date": predict_date,
                                "data_warning": data_warning
                            }
                except Exception as e:
                    # Log errors for debugging
                    import traceback
                    st.sidebar.error(f"❌ Error predicting {district_name}: {str(e)}")
                    with st.sidebar.expander("Debug Info"):
                        st.code(traceback.format_exc())
                
                progress_bar.progress((idx + 1) / len(districts_in_province))
            
            progress_bar.empty()
        
        st.sidebar.success(f"✅ Predicted {len(predicted_ndvi_dict)} districts")
        
        # Show missing data info if any
        total_districts = len(districts_in_province)
        missing_count = total_districts - len(predicted_ndvi_dict)
        if missing_count > 0:
            st.sidebar.warning(f"⚠️ {missing_count} districts missing data")
            with st.sidebar.expander("Why is data missing?"):
                st.markdown("""
                **Possible reasons:**
                - Insufficient historical NDVI data (< 3 periods)
                - Incomplete ERA5 weather data (< 16 days)
                - Cloud coverage affecting satellite imagery
                - Recently established administrative areas
                - Data collection limitations in specific regions
                """)
    
    if not selected_province:
        st.info("👈 Select a province to see NDVI maps and predictions")
        
        # Show all provinces map
        map_obj = create_map(geojson_province, geojson_district, None, None, "Vietnam Provinces", location_df=location_df)
        st_folium(map_obj, width=None, height=600, use_container_width=True)
    
    else:
        # Determine highlighted district (for later use in maps)
        highlighted_gid2 = None
        selected_detail_district = None
        
        # 1. NDVI Maps
        st.header("NDVI Maps")
        
        # Show maps and legend in 3 columns
        col_map1, col_map2, col_legend = st.columns([1, 1, 0.15])
        
        with col_map1:
            st.subheader("Current NDVI")
            st.markdown("<p style='color:#fff;font-size:12px;margin-top:-10px;'>Latest 16-day composite</p>", unsafe_allow_html=True)
            map_current = create_map(geojson_province, geojson_district, gid1, current_ndvi_dict, "Current NDVI", highlighted_gid2, location_df)
            st_folium(map_current, width=None, height=500, use_container_width=True, returned_objects=[], key="map_current")
        
        with col_map2:
            st.subheader("Predicted NDVI")
            st.markdown("<p style='color:#fff;font-size:12px;margin-top:-10px;'>Next 16-day forecast</p>", unsafe_allow_html=True)
            map_predicted = create_map(geojson_province, geojson_district, gid1, predicted_ndvi_dict, "Predicted NDVI", highlighted_gid2, location_df)
            st_folium(map_predicted, width=None, height=500, use_container_width=True, returned_objects=[], key="map_predicted")
        
        with col_legend:
            create_ndvi_legend()
        
        # 2. Province Summary
        if predicted_ndvi_dict:
            st.divider()
            st.subheader("Province Summary")
            
            current_vals = list(current_ndvi_dict.values())
            predicted_vals = list(predicted_ndvi_dict.values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Districts Analyzed", len(predicted_ndvi_dict))
            
            with col2:
                if current_vals:
                    avg_current = np.mean(current_vals)
                    st.metric("Avg Current NDVI", f"{avg_current:.3f}")
                else:
                    st.metric("Avg Current NDVI", "N/A")
            
            with col3:
                avg_predicted = np.mean(predicted_vals)
                st.metric("Avg Predicted NDVI", f"{avg_predicted:.3f}")
            
            with col4:
                if current_vals:
                    change = avg_predicted - avg_current
                    st.metric("Expected Change", f"{change:+.3f}")
                else:
                    st.metric("Expected Change", "N/A")
            
            # Count missing predictions
            total_districts = len(districts_in_province)
            successful_predictions = len(predicted_ndvi_dict)
            missing_count = total_districts - successful_predictions
            
            if missing_count > 0:
                st.metric("⚠️ Missing Data", missing_count)
            
            st.divider()
            st.subheader("District Comparison")
            
            # Prepare data for heatmap
            district_names = [prediction_details[gid2]["name"] for gid2 in prediction_details.keys()]
            current_ndvi_vals = [prediction_details[gid2]["current_ndvi"] for gid2 in prediction_details.keys()]
            predicted_ndvi_vals = [prediction_details[gid2]["predicted_ndvi"] for gid2 in prediction_details.keys()]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current NDVI',
                x=district_names,
                y=current_ndvi_vals,
                marker_color='lightblue',
                text=[f"{v:.3f}" for v in current_ndvi_vals],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Predicted NDVI',
                x=district_names,
                y=predicted_ndvi_vals,
                marker_color='lightgreen',
                text=[f"{v:.3f}" for v in predicted_ndvi_vals],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"NDVI Comparison Across Districts in {selected_province}",
                xaxis_title="District",
                yaxis_title="NDVI",
                barmode='group',
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. District Details
            st.divider()
            st.header("District Details")
            
            # Get list of predicted districts
            district_options = [""] + [prediction_details[gid2]["name"] for gid2 in prediction_details.keys()]
            selected_detail_district = st.selectbox(
                "Select a district to view detailed information:",
                options=district_options,
                format_func=lambda x: "-- Select District --" if x == "" else x
            )
        
        # 5. Show district detail card if selected
        if predicted_ndvi_dict and selected_detail_district and selected_detail_district != "":
            # Find GID_2 for selected district name
            selected_gid2 = None
            for gid2, details in prediction_details.items():
                if details["name"] == selected_detail_district:
                    selected_gid2 = gid2
                    break
            
            if selected_gid2 and selected_gid2 in prediction_details:
                details = prediction_details[selected_gid2]
                
                st.divider()
                st.divider()
                st.markdown(f"### {details['name']}")
                
                # Show warning if district has incomplete weather data
                if details.get('era5_window') is not None:
                    era5_days = len(details['era5_window'])
                    if era5_days < 16:
                        st.warning(f"⚠️ **Incomplete Weather Data:** Only {era5_days}/16 days of ERA5 data available for this prediction. Results may be less accurate.")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current NDVI",
                        f"{details['current_ndvi']:.4f}"
                    )
                with col2:
                    st.metric(
                        "Predicted NDVI",
                        f"{details['predicted_ndvi']:.4f}",
                        delta=f"{details['predicted_ndvi'] - details['current_ndvi']:.4f}"
                    )
                with col3:
                    category = get_ndvi_category(details['predicted_ndvi'])
                    color = get_ndvi_color(details['predicted_ndvi'])
                    st.markdown(f"""
                    <div style='padding:15px;background-color:{color};border-radius:10px;text-align:center;display:flex;align-items:center;justify-content:center;height:100%;'>
                        <h4 style='color:white;margin:0;text-shadow: 1px 1px 2px black;'>{category}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show data quality warning if present
                if details.get("data_warning"):
                    st.warning(f"⚠️ **Data Quality:** {details['data_warning']}")
                
                # Meteorological data
                if details['era5_window'] is not None and len(details['era5_window']) > 0:
                        st.markdown("")
                        st.markdown("**Input Weather Data**")
                        
                        era5_window = details['era5_window']
                        window_start = era5_window["date"].min()
                        window_end = era5_window["date"].max()
                        
                        # Calculate predicted period
                        predicted_start = details['predict_date']
                        predicted_end = predicted_start + timedelta(days=15)
                        
                        st.caption(f"""
                        **Input Period:** {window_start.strftime('%Y-%m-%d')} → {window_end.strftime('%Y-%m-%d')} ({len(era5_window)} days)  
                        **Predicted NDVI Period:** {predicted_start.strftime('%Y-%m-%d')} → {predicted_end.strftime('%Y-%m-%d')} (16-day composite)
                        """)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🌧️ Total Precip", f"{era5_window['total_precipitation_sum'].sum():.1f} mm")
                        with col2:
                            st.metric("🌡️ Avg Temp", f"{era5_window['2m_temperature_mean'].mean():.1f} °C")
                        with col3:
                            st.metric("☀️ Avg Solar", f"{era5_window['surface_solar_radiation_downwards_sum'].mean()/1e6:.1f} MJ/m²")
                        with col4:
                            st.metric("💧 Avg Soil Moisture", f"{era5_window['volumetric_soil_water_layer_1_mean'].mean():.3f} m³/m³")
                        
                        # Weather plots
                        st.markdown("**📈 Weather Trends**")
                        
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=("Daily Precipitation", "Temperature", "Solar Radiation", "Soil Moisture"),
                            vertical_spacing=0.18,  # Increased from 0.12
                            horizontal_spacing=0.12  # Increased from 0.1
                        )
                        
                        dates = era5_window["date"]
                        
                        # Precipitation
                        fig.add_trace(
                            go.Bar(x=dates, y=era5_window["total_precipitation_sum"], name="Precip", marker_color="#1f77b4"),
                            row=1, col=1
                        )
                        fig.update_yaxes(title_text="mm", row=1, col=1)
                        
                        # Temperature
                        fig.add_trace(
                            go.Scatter(x=dates, y=era5_window["2m_temperature_mean"], mode="lines+markers", name="Temp", line=dict(color="#ff7f0e")),
                            row=1, col=2
                        )
                        fig.update_yaxes(title_text="°C", row=1, col=2)
                        
                        # Solar Radiation
                        fig.add_trace(
                            go.Scatter(x=dates, y=era5_window["surface_solar_radiation_downwards_sum"]/1e6, mode="lines+markers", name="Solar", line=dict(color="#2ca02c")),
                            row=2, col=1
                        )
                        fig.update_yaxes(title_text="MJ/m²", row=2, col=1)
                        
                        # Soil Moisture
                        fig.add_trace(
                            go.Scatter(x=dates, y=era5_window["volumetric_soil_water_layer_1_mean"], mode="lines+markers", name="SM Layer 1", line=dict(color="#d62728")),
                            row=2, col=2
                        )
                        if "volumetric_soil_water_layer_2_mean" in era5_window.columns:
                            fig.add_trace(
                                go.Scatter(x=dates, y=era5_window["volumetric_soil_water_layer_2_mean"], mode="lines+markers", name="SM Layer 2", line=dict(color="#9467bd", dash="dash")),
                                row=2, col=2
                            )
                        fig.update_yaxes(title_text="m³/m³", row=2, col=2)
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:gray;'>
        <p>🌿 NDVI Prediction System for Mekong Delta | Built with Streamlit & Keras</p>
        <p>Data sources: ERA5-Land (Climate) & MODIS MOD13Q1 (NDVI)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
