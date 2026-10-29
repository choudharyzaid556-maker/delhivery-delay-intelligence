from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[[
        "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
        "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
        "hub_tier", "hub_daily_capacity",
        "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
    ]]
    y = df["delivery_hours"]

    temp = df.copy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    Path("outputs").mkdir(exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_scaled_max_abs_mean = np.max(np.abs(X_train_scaled.mean(axis=0)))
    test_scaled_max_abs_mean = np.max(np.abs(X_test_scaled.mean(axis=0)))

    pd.DataFrame([
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {"metric": "train_scaled_max_abs_mean", "value": float(train_scaled_max_abs_mean)},
        {"metric": "test_scaled_max_abs_mean", "value": float(test_scaled_max_abs_mean)},
    ]).to_csv("outputs/part_b_checks.csv", index=False)

    baseline_pred = np.full(len(y_val), y_train.mean())
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    baseline_mse = mean_squared_error(y_val, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_val, baseline_pred)

    tree_rows = []
    for d in [1, 2, 3, 4, 6, 8, 12]:
        model = DecisionTreeRegressor(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        tr = model.predict(X_train)
        vr = model.predict(X_val)
        tree_rows.append({
            "max_depth": d,
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, tr))),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, vr))),
        })
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    tr = model.predict(X_train)
    vr = model.predict(X_val)
    tree_rows.append({
        "max_depth": "unlimited",
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, tr))),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, vr))),
    })
    pd.DataFrame(tree_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

    model_specs = {
        "mean_baseline": None,
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = []
    for name, model in model_specs.items():
        if name == "mean_baseline":
            pred = baseline_pred
            mae = baseline_mae
            mse = baseline_mse
            rmse = baseline_rmse
            r2 = baseline_r2
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            mse = mean_squared_error(y_val, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, pred)
        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
        })
    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)

    gb_rows = []
    for lr in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        gb_rows.append({
            "learning_rate": lr,
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
            "val_r2": float(r2_score(y_val, pred)),
        })
    pd.DataFrame(gb_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    candidates = [
        ("linear_regression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
        ("decision_tree", DecisionTreeRegressor(random_state=42)),
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name, model in candidates:
        model.fit(X_train, y_train)
        scores = cross_val_score(model, pd.concat([X_train, X_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True), cv=kf)
        cv_rows.append({"model": name, "mean_r2": float(scores.mean()), "std_r2": float(scores.std())})
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

    # final model selected on validation RMSE
    best_name = min(["linear_regression", "decision_tree", "random_forest", "gradient_boosting"], key=lambda n: r2_score(y_val, {"linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), "decision_tree": DecisionTreeRegressor(random_state=42), "random_forest": RandomForestRegressor(n_estimators=100, random_state=42), "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)}[n].fit(X_train, y_train).predict(X_val)))
    final_model = {"linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), "decision_tree": DecisionTreeRegressor(random_state=42), "random_forest": RandomForestRegressor(n_estimators=100, random_state=42), "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)}[best_name]
    final_model.fit(X_train, y_train)
    pred_test = final_model.predict(X_test)
    final_row = {
        "model": best_name,
        "split": "test",
        "MAE": float(mean_absolute_error(y_test, pred_test)),
        "MSE": float(mean_squared_error(y_test, pred_test)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "R2": float(r2_score(y_test, pred_test)),
    }
    metrics = pd.read_csv("outputs/part_b_metrics.csv")
    metrics = pd.concat([metrics, pd.DataFrame([final_row])], ignore_index=True)
    metrics.to_csv("outputs/part_b_metrics.csv", index=False)


if __name__ == "__main__":
    main()
