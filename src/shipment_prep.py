import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Clean and standardize a raw shipment table."""
    df = raw.copy().drop_duplicates().copy()
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.drop(columns=["declared_value"], errors="ignore")
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce")
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {}
        for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
            medians[col] = float(df[col].median())
    else:
        medians = {"distance_km": float(medians["distance_km"]), "parcel_weight_kg": float(medians["parcel_weight_kg"]), "declared_value_inr": float(medians["declared_value_inr"])}

    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
        df[col] = df[col].fillna(medians[col])
    return df, medians


def build_features(clean, hubs):
    """Attach hub metadata and derived feature columns."""
    df = clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = df["payment_mode"].eq("COD").astype(int)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
