from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    modelling = pd.read_csv("data/modelling_table.csv")
    hub_profile = (
        modelling.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    hub_profile.to_csv("outputs/hub_profile.csv", index=False)

    profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaled = StandardScaler().fit_transform(hub_profile[profile_cols])

    separation_rows = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        same = []
        diff = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    same.append(np.linalg.norm(scaled[i] - scaled[j]))
                else:
                    diff.append(np.linalg.norm(scaled[i] - scaled[j]))
        ratio = float(np.mean(diff) / np.mean(same)) if np.mean(same) else np.nan
        cluster_sizes = "-".join(str(x) for x in sorted(np.bincount(labels), reverse=True))
        separation_rows.append({"K": k, "across_over_within": ratio, "cluster_sizes": cluster_sizes})
    pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

    hub_profile["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    cluster_summary = hub_profile.groupby("cluster").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).reset_index()
    cluster_summary = cluster_summary.sort_values("delay_rate", ascending=True).reset_index(drop=True)
    cluster_summary["segment"] = ["Standard", "Watch", "Escalate"]
    segment_map = dict(zip(cluster_summary["cluster"], cluster_summary["segment"]))
    hub_profile["segment"] = hub_profile["cluster"].map(segment_map)

    hub_segments = hub_profile[["hub_id", "segment"]].merge(pd.read_csv("data/hubs.csv"), on="hub_id", how="left")
    hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    seg_profile = cluster_summary[["segment", "hubs", "mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)

    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatch = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(tier_map)].copy()
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
