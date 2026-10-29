from pathlib import Path
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS


def main():
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    clean_df, medians = clean_shipments(raw)
    model_df = build_features(clean_df, hubs)
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

    # row count check
    model_df.to_csv("data/modelling_table.csv", index=False)

    # checks file
    before = raw.drop_duplicates().copy()
    after = clean_df.copy()
    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(before),
        "rows_after_dedup": len(before),
        "missing_distance_km_before": int(before["distance_km"].isna().sum()),
        "missing_parcel_weight_before": int(before["parcel_weight_kg"].isna().sum() - before["parcel_weight_kg"].eq(999.0).sum()),
        "sentinel_weight_rows": int(before["parcel_weight_kg"].eq(999.0).sum()),
        "missing_declared_value_before": int(before["declared_value"].isna().sum()),
        "missing_payment_mode_before": int(before["payment_mode"].isna().sum()),
        "service_type_spellings_before": int(before["service_type"].nunique()),
        "service_type_spellings_after": int(after["service_type"].nunique()),
        "median_distance_km": float(model_df["distance_km"].median()),
        "median_parcel_weight_kg": float(model_df["parcel_weight_kg"].median()),
        "median_declared_value_inr": float(model_df["declared_value_inr"].median()),
        "rows_after_hub_merge": len(model_df),
        "nulls_in_features_and_target": int(model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        "delayed_count": int(model_df["is_delayed"].sum()),
        "delay_rate": float(model_df["is_delayed"].mean()),
        "hours_q1": float(model_df["delivery_hours"].quantile(0.25)),
        "hours_q3": float(model_df["delivery_hours"].quantile(0.75)),
        "declared_value_mean_inr": float(model_df["declared_value_inr"].mean()),
        "declared_value_median_inr": float(model_df["declared_value_inr"].median()),
        "log_declared_value_mean": float(np.log(model_df["declared_value_inr"]).mean()),
        "log_declared_value_median": float(np.log(model_df["declared_value_inr"]).median()),
    }
    q1 = model_df["delivery_hours"].quantile(0.25)
    q3 = model_df["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    checks["hours_lower_bound"] = float(lower)
    checks["hours_upper_bound"] = float(upper)
    checks["hours_outlier_count"] = int(((model_df["delivery_hours"] < lower) | (model_df["delivery_hours"] > upper)).sum())

    rows = [{"metric": k, "value": v} for k, v in checks.items()]
    pd.DataFrame(rows).to_csv("outputs/part_a_checks.csv", index=False)

    # delay trends by group
    def delay_group(df, by):
        out = (
            df.groupby(by, dropna=False)
            .agg(shipments=("shipment_id", "size"), delayed=("is_delayed", "sum"), mean_hours=("delivery_hours", "mean"))
            .reset_index()
        )
        out["delay_rate"] = out["delayed"] / out["shipments"]
        return out

    service = delay_group(model_df, "service_type")
    service["dimension"] = "service_type"
    service = service[["dimension", "service_type", "shipments", "delayed", "delay_rate", "mean_hours"]].rename(columns={"service_type": "group"})

    tier = delay_group(model_df, "hub_tier")
    tier["dimension"] = "hub_tier"
    tier = tier[["dimension", "hub_tier", "shipments", "delayed", "delay_rate", "mean_hours"]].rename(columns={"hub_tier": "group"})

    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        include_lowest=True,
        right=True,
    )
    dist = delay_group(model_df, "distance_band")
    dist["dimension"] = "distance_band"
    dist = dist[["dimension", "distance_band", "shipments", "delayed", "delay_rate", "mean_hours"]].rename(columns={"distance_band": "group"})

    group_df = pd.concat([service, tier, dist], ignore_index=True)
    cols = ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
    group_df = group_df[cols]
    group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    corr = model_df.select_dtypes(include=[np.number]).corr().reset_index().rename(columns={"index": "column"})
    corr.to_csv("outputs/part_a_correlations.csv", index=False)

    Path("outputs").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
