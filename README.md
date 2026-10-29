# Capstone Project — Delhivery Last-Mile Delay Intelligence Pipeline

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl` and the associated `models/*.csv` files. The model artifact is intentionally tied to the scikit-learn version pinned in `requirements.txt` and should be rebuilt whenever the environment changes.

## Data findings

- The most delayed service type is Express at a rate of 0.4897, compared with Standard at 0.2056 and Economy at 0.0857.
- Delay rate rises sharply from Tier 1 to Tier 3: 0.0400 in Tier 1, 0.2150 in Tier 2 and 0.4400 in Tier 3.
- Distance is strongly linked to lateness: the delay rate rises from 0.0219 in the 0-200 km band to 0.1368 in 201-400 km, 0.4364 in 401-600 km and 0.8523 in 601+ km.
- The three strongest correlates with `delivery_hours` are `promised_hours` (0.8370), `distance_km` (0.4010) and `is_delayed` (0.5460) when ranked by absolute correlation excluding `delivery_hours` itself. In the output file, the signed correlations are preserved to four decimals.
- `declared_value_inr` was log-transformed because its distribution is strongly right-skewed: the filled median is 1121.50 and the mean is 1533.05; after the log transform the median is 7.0220 and the mean is 7.0280, giving a much more stable scale for modelling.

## Model results (regression)

The model comparison used a leakage-safe train/validation/test split and evaluated multiple regressors. The baseline validation R² is −0.021, which means a constant prediction does not explain the parcel-to-parcel variation in delivery hours. The winner on validation was the model with the lowest RMSE, and the tree-depth diagnostic showed the standard bias–variance pattern: very shallow trees are high-bias and a depth-unlimited tree overfits the training set with a large train-vs-validation gap. The learning-rate experiment identified 0.1 as the best gradient-boosting setting because 0.01 was too slow, while 1.0 overshot the optimum and reduced validation performance.

## Model results (classification)

The blind baseline predicts every parcel is on time and therefore achieves 0.7667 validation accuracy while being operationally useless. The noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) account for a small share of forest importance (about 0.225), and all five rank outside the top 7 by F-score. The feature-selection table shows that the strongest signal comes from distance, hub profile, stops and promise characteristics, and `k = 7` gave the best validation balance before tuning. Tuning improved the validation F1 relative to the untuned random forest, but the gap is modest on a 180-row validation split, so tuning should be treated as a moderate improvement rather than a dramatic upgrade. The chosen alert threshold is driven by the operations rule: keep the highest threshold whose validation recall is at least 0.80, balancing recall with precision. On the test set the final model achieved precision about 0.689, recall about 0.756 and F1 about 0.721.

## Operations recommendation

The regional operations team should prioritize the `Escalate` segment, which has the highest delay rate, and use the tier labels only after checking the actual segment assignments. The final rule combines probability-based alerting with the hub segment: flagged parcels in `Escalate` hubs are escalated to the hub manager, flagged parcels in other segments receive a proactive customer call, and unflagged parcels receive no action. In the hub mismatch table, `HUB07` (Tier 1 but `Watch`), `HUB13` (Tier 2 but `Escalate`), `HUB15` (Tier 2 but `Escalate`) and `HUB22` (Tier 3 but `Watch`) all show that tier labels are not sufficient on their own for dispatch decisions. We keep `K = 3` because it is the smallest segmentation that still separates the operational playbooks while remaining stable across random seeds and hub orderings; `K = 4` and `K = 5` create tiny fragmentation and therefore weak operational action groups.

## Production plan

Production readiness is mapped to concrete repository elements: reproducibility is ensured by the pinned package versions in `requirements.txt` and the script ordering in the README; reliability comes from the shared cleaning pipeline in `src/shipment_prep.py`; maintainability is improved by the separate stage scripts (`part_a_clean_eda.py`, `part_b_regression.py`, `part_c_classification.py`, `part_d_segments.py`); performance is acceptable because all features are precomputed for inference; consistency is guaranteed by using the same preprocessing logic for training and prediction; and monitorability is supported by the outputs and CSV artefacts in `outputs/` and `models/`.

Key metrics to monitor with numeric triggers include: weekly delayed share of delivered parcels (trigger: rising above 5 percentage points versus the trailing 8-week baseline), share of parcels flagged (trigger: more than 20% weekly flagged volume or a sustained > 10% increase), and recall measured on parcels whose outcome is known (trigger: recall falling below 0.80 on the latest 30-day scored batch). Retraining should happen monthly, with a priority recheck immediately after any major network change. The 15–28 September sale interval should always be compared against comparable sale weeks rather than ordinary weeks, because sale periods are structurally different from baseline operations. Data drift would look like a rising share of late parcels, a different hub-segment mix, or a systematic shift in feature distributions like heavier parcels, longer distances, or more weekend dispatches.

## Limitations

- The data are synthetic, so the patterns are useful for a capstone exercise but not a direct replica of Delhivery's production network.
- The test set contains only 41 delayed parcels, so test recall and precision are naturally noisy despite the strong operational logic.
- The alert threshold is tuned on a single validation split, so different random splits could lead to a slightly different threshold.
- Hub-level segmentation is intentionally simple and uses only a small set of operational profiles, which may miss more granular route dynamics.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   ├── __init__.py
│   └── shipment_prep.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_a_correlations.csv
│   └── ...
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
└── .gitignore
```
