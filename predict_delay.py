import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features


MODELS = "models"


def main():
    raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    medians = {"distance_km": 314.95, "parcel_weight_kg": 1.83, "declared_value_inr": 1121.5}
    cleaned, _ = clean_shipments(raw, medians=medians)
    features = build_features(cleaned, hubs)
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    X = features[feature_order]
    model = __import__("joblib").load("models/delay_model.pkl")
    proba = model.predict_proba(X)[:, 1]
    results = []
    threshold = float(pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"])
    for row, p in zip(raw.to_dict(orient="records"), proba):
        shipment_id = row["shipment_id"]
        hub_id = row["hub_id"]
        segment = pd.read_csv("outputs/hub_segments.csv").set_index("hub_id").loc[hub_id, "segment"]
        flagged = int(p >= threshold)
        if flagged and segment == "Escalate":
            action = "Escalate to hub manager"
        elif flagged:
            action = "Proactive customer call"
        else:
            action = "No action"
        results.append({
            "shipment_id": shipment_id,
            "hub_id": hub_id,
            "segment": segment,
            "delay_probability": round(float(p), 4),
            "flagged": flagged,
            "action": action,
        })
    pd.DataFrame(results).to_csv("outputs/new_shipment_actions.csv", index=False)


if __name__ == "__main__":
    main()
