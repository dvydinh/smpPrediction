import re


TARGET_COLUMN = "smp_system_price"

OBSERVED_ONLY_COLUMNS = {
    "load_total_mw",
    "load_north_mw",
    "load_central_mw",
    "load_south_mw",
    "hydro_inflow_m3s",
    "hydro_total_discharge_m3s",
    "hydro_plant_discharge_m3s",
    "hydro_spill_discharge_m3s",
    "hydro_water_level_m",
    "coal_proxy_price",
    "brent_price",
    "gas_proxy_price",
    "usd_vnd",
    "dxy_index",
}

RAW_REGIONAL_PRICE_COLUMNS = {
    "smp_north_price",
    "smp_central_price",
    "smp_south_price",
}

WEATHER_PREFIXES = (
    "temperature_",
    "humidity_",
    "cloud_cover_",
    "wind_speed_",
    "shortwave_radiation_",
    "direct_radiation_",
    "diffuse_radiation_",
)

WEATHER_DERIVED_COLUMNS = {
    "solar_gen_proxy",
    "wind_gen_proxy",
    "residual_load_proxy",
    "thermal_margin_proxy",
    "load_to_rad_ratio",
    "load_to_wind_ratio",
    "heat_stress_hn",
    "cold_stress_hn",
    "heat_stress_hcmc",
    "precip_total",
    "precip_rolling_7d",
    "precip_rolling_30d",
    "hydro_stress_proxy",
}


def _unavailable_at_origin(col):
    if col in OBSERVED_ONLY_COLUMNS | RAW_REGIONAL_PRICE_COLUMNS | {TARGET_COLUMN, "datetime"}:
        return True
    if col in WEATHER_DERIVED_COLUMNS:
        return True
    if col.startswith(WEATHER_PREFIXES) and "_same_cycle_" not in col:
        return True
    if col.startswith("disp_") and "_same_cycle_" not in col:
        return True
    return False


def select_production_features(df):
    """Return numeric features available by 08:00 on the forecast origin day."""
    features = []
    for col in df.select_dtypes(include="number").columns:
        if _unavailable_at_origin(col):
            continue
        if df[col].notna().mean() < 0.95:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        features.append(col)
    validate_production_features(features)
    return features


def validate_production_features(feature_cols):
    blocked = {col for col in feature_cols if _unavailable_at_origin(col)}
    unsafe_raw_prices = {
        col for col in feature_cols
        if re.fullmatch(r"smp_(system|north|central|south)_price", col)
    }
    blocked.update(unsafe_raw_prices)
    if blocked:
        names = ", ".join(sorted(blocked))
        raise ValueError(f"Unavailable 08:00 features selected: {names}")
