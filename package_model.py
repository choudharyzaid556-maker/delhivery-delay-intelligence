import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS


def main():
    # re-create the same split used in Part C
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    selector = SelectKBest(f_classif, k=7)
    X_train_sel = selector.fit_transform(X_train, y_train)
    model = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_leaf=1)
    pipeline = Pipeline([("selector", selector), ("model", model)])
    pipeline.fit(X_train, y_train)

    # save model files
    joblib.dump(pipeline, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [0.5]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    medians = {"distance_km": float(df["distance_km"].median()), "parcel_weight_kg": float(df["parcel_weight_kg"].median()), "declared_value_inr": float(df["declared_value_inr"].median())}
    pd.DataFrame({"column": ["distance_km", "parcel_weight_kg", "declared_value_inr"], "median": [medians["distance_km"], medians["parcel_weight_kg"], medians["declared_value_inr"]]}).to_csv("models/impute_values.csv", index=False)

    reloaded = joblib.load("models/delay_model.pkl")
    prob_reload = reloaded.predict_proba(X_test)[:, 1]
    prob_inline = pipeline.predict_proba(X_test)[:, 1]
    diff = abs(prob_reload - prob_inline).max()
    pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"], "value": [len(X_test), float(diff), 0.5]}).to_csv("outputs/part_d_reload_check.csv", index=False)


if __name__ == "__main__":
    main()
