from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def score_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return tn, fp, fn, tp, acc, prec, rec, f1


def main():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    Path("outputs").mkdir(exist_ok=True)

    blind = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp, acc, prec, rec, f1 = score_model(y_val, blind)
    pd.DataFrame([{
        "model": "blind_all_on_time",
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }]).to_csv("outputs/part_c_models.csv", index=False)

    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
    }
    for d in [2, 3, 4, 5, 6, 8, 10]:
        models[f"decision_tree_depth_{d}"] = DecisionTreeClassifier(max_depth=d, random_state=42)
    models["random_forest_default"] = RandomForestClassifier(n_estimators=100, random_state=42)

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        tn, fp, fn, tp, acc, prec, rec, f1 = score_model(y_val, pred)
        rows.append({
            "model": name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })
    pd.DataFrame(rows).to_csv("outputs/part_c_models.csv", index=False)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    selector = SelectKBest(f_classif, k="all")
    selector.fit(X_train, y_train)
    scores = selector.scores_
    feature_scores = []
    for feature, score, importance in zip(FEATURES, scores, importances):
        feature_scores.append({
            "feature": feature,
            "f_score": float(score),
            "f_rank": 0,
            "rf_importance": float(importance),
            "is_noise": feature in NOISE_COLUMNS,
        })
    feature_scores = pd.DataFrame(feature_scores).sort_values("f_score", ascending=False).reset_index(drop=True)
    feature_scores["f_rank"] = np.arange(1, len(feature_scores) + 1)
    feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        selectk_rows.append({"k": k, "val_f1": float(f1_score(y_val, pred))})
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

    selector7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector7.fit_transform(X_train, y_train)
    X_val_7 = selector7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train_7, y_train)
    grid_pred = grid.best_estimator_.predict(X_val_7)
    grid_f1 = f1_score(y_val, grid_pred)

    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions={"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}, n_iter=9, cv=5, random_state=42)
    rand.fit(X_train_7, y_train)
    rand_pred = rand.best_estimator_.predict(X_val_7)
    rand_f1 = f1_score(y_val, rand_pred)

    default_model = RandomForestClassifier(n_estimators=100, random_state=42)
    default_model.fit(X_train_7, y_train)
    default_pred = default_model.predict(X_val_7)
    default_f1 = f1_score(y_val, default_pred)

    tuning = pd.DataFrame([
        {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135, "best_params": str(grid.best_params_), "best_cv_accuracy": float(grid.best_score_), "val_f1": float(grid_f1)},
        {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45, "best_params": str(rand.best_params_), "best_cv_accuracy": float(rand.best_score_), "val_f1": float(rand_f1)},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1, "best_params": "defaults", "best_cv_accuracy": "", "val_f1": float(default_f1)},
    ])
    tuning.to_csv("outputs/part_c_tuning.csv", index=False)

    # choose winner by validation F1, tie-breaking as specified
    best_name = "default_forest_same_7_features" if default_f1 >= max(grid_f1, rand_f1) else ("grid_search" if grid_f1 >= rand_f1 else "randomized_search")
    chosen = grid if best_name == "grid_search" else rand if best_name == "randomized_search" else None
    if best_name == "default_forest_same_7_features":
        final_model = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        final_model.fit(X_train, y_train)
        params = {"selector_k": 7, "n_estimators": 100, "max_depth": "unlimited", "min_samples_leaf": 1, "threshold": 0.5, "chosen_from": best_name}
    else:
        final_model = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", RandomForestClassifier(random_state=42, **chosen.best_params_))])
        final_model.fit(X_train, y_train)
        params = {"selector_k": 7, "n_estimators": chosen.best_params_["n_estimators"], "max_depth": chosen.best_params_.get("max_depth", "unlimited"), "min_samples_leaf": chosen.best_params_["min_samples_leaf"], "threshold": 0.5, "chosen_from": best_name}

    val_proba = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_rows = []
    chosen_threshold = 0.1
    for threshold in thresholds:
        pred = (val_proba >= threshold).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = score_model(y_val, pred)
        threshold_rows.append({"threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        if rec >= 0.80:
            chosen_threshold = threshold
    pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    pred_test = (test_proba >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = score_model(y_test, pred_test)
    pd.DataFrame([{"threshold": chosen_threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}]).to_csv("outputs/part_c_final_test.csv", index=False)
    pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": params["chosen_from"],
    }]).to_csv("outputs/part_c_final_config.csv", index=False)


if __name__ == "__main__":
    main()
