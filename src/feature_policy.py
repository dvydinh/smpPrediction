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


def select_production_features(df):
    """Return numeric features available by 08:00 on the forecast origin day.

    Weather columns are allowed only under the contract that backtests contain the
    forecast vintage that would have been available at the forecast origin.
    """
    blocked = OBSERVED_ONLY_COLUMNS | RAW_REGIONAL_PRICE_COLUMNS | {TARGET_COLUMN, "datetime"}
    features = []
    for col in df.select_dtypes(include="number").columns:
        if col in blocked:
            continue
        if df[col].notna().mean() < 0.95:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        features.append(col)
    validate_production_features(features)
    return features


def validate_production_features(feature_cols):
    blocked = (OBSERVED_ONLY_COLUMNS | RAW_REGIONAL_PRICE_COLUMNS).intersection(feature_cols)
    unsafe_raw_prices = {
        col for col in feature_cols
        if re.fullmatch(r"smp_(system|north|central|south)_price", col)
    }
    blocked.update(unsafe_raw_prices)
    if blocked:
        names = ", ".join(sorted(blocked))
        raise ValueError(f"Unavailable 08:00 features selected: {names}")
