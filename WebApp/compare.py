import os
import time
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "sleep_fatigue_model.pkl"
ONNX_PATH = "sleep_fatigue.onnx"

# Change this if your CSV has a different filename
DATA_PATH = "bedtime_screentime_sleep_debt.csv"

TARGET = "next_day_fatigue_score"

FEATURES = [
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
    "morning_alarm_snoozes",
    "gender",
    "occupation_type",
    "chronotype",
    "primary_bedtime_app",
    "blue_light_filter_active",
]

NUMERICAL_FEATURES = [
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
    "morning_alarm_snoozes",
]

CATEGORICAL_FEATURES = [
    "gender",
    "occupation_type",
    "chronotype",
    "primary_bedtime_app",
]

INTEGER_FEATURES = [
    "blue_light_filter_active",
]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"Missing ONNX model: {ONNX_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Change DATA_PATH at the top of this script."
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

sklearn_model = joblib.load(MODEL_PATH)

ort_session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

print("Sklearn model loaded.")
print("ONNX model loaded.")

print("\nONNX INPUTS:")
for input_node in ort_session.get_inputs():
    print(
        f"  {input_node.name:30} "
        f"type={input_node.type:15} "
        f"shape={input_node.shape}"
    )

print("\nONNX OUTPUTS:")
for output_node in ort_session.get_outputs():
    print(
        f"  {output_node.name:30} "
        f"type={output_node.type:15} "
        f"shape={output_node.shape}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

missing = [column for column in FEATURES + [TARGET] if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


X = df[FEATURES].copy()
y = df[TARGET].astype(np.float32).to_numpy()


# ============================================================
# SKLEARN PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("SKLEARN PREDICTION")
print("=" * 60)

start = time.perf_counter()

sklearn_predictions = sklearn_model.predict(X)

sklearn_time = time.perf_counter() - start

sklearn_predictions = np.asarray(
    sklearn_predictions,
    dtype=np.float32
).reshape(-1)


# ============================================================
# PREPARE ONNX INPUTS
# ============================================================

onnx_inputs = {}

for feature in NUMERICAL_FEATURES:
    onnx_inputs[feature] = (
        X[feature]
        .astype(np.float32)
        .to_numpy()
        .reshape(-1, 1)
    )

for feature in CATEGORICAL_FEATURES:
    onnx_inputs[feature] = (
        X[feature]
        .astype(str)
        .to_numpy()
        .reshape(-1, 1)
    )

onnx_inputs["blue_light_filter_active"] = (
    X["blue_light_filter_active"]
    .astype(np.int64)
    .to_numpy()
    .reshape(-1, 1)
)


# ============================================================
# ONNX PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("ONNX PREDICTION")
print("=" * 60)

start = time.perf_counter()

onnx_result = ort_session.run(
    None,
    onnx_inputs
)

onnx_time = time.perf_counter() - start

onnx_predictions = np.asarray(
    onnx_result[0],
    dtype=np.float32
).reshape(-1)


# ============================================================
# PREDICTION COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION EQUIVALENCE")
print("=" * 60)

absolute_difference = np.abs(
    sklearn_predictions - onnx_predictions
)

print(f"Maximum absolute difference : {absolute_difference.max():.10f}")
print(f"Mean absolute difference    : {absolute_difference.mean():.10f}")
print(f"Median absolute difference  : {np.median(absolute_difference):.10f}")


# ============================================================
# PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

sklearn_mae = mean_absolute_error(
    y,
    sklearn_predictions
)

sklearn_rmse = np.sqrt(
    mean_squared_error(
        y,
        sklearn_predictions
    )
)

sklearn_r2 = r2_score(
    y,
    sklearn_predictions
)


onnx_mae = mean_absolute_error(
    y,
    onnx_predictions
)

onnx_rmse = np.sqrt(
    mean_squared_error(
        y,
        onnx_predictions
    )
)

onnx_r2 = r2_score(
    y,
    onnx_predictions
)


print("\nSklearn (.pkl)")
print(f"MAE  : {sklearn_mae:.6f}")
print(f"RMSE : {sklearn_rmse:.6f}")
print(f"R²   : {sklearn_r2:.6f}")

print("\nONNX")
print(f"MAE  : {onnx_mae:.6f}")
print(f"RMSE : {onnx_rmse:.6f}")
print(f"R²   : {onnx_r2:.6f}")


# ============================================================
# LATENCY
# ============================================================

print("\n" + "=" * 60)
print("INFERENCE PERFORMANCE")
print("=" * 60)

print(f"Sklearn total time : {sklearn_time:.6f} seconds")
print(f"ONNX total time    : {onnx_time:.6f} seconds")

print(
    f"\nSklearn per sample : "
    f"{(sklearn_time / len(X)) * 1000:.4f} ms"
)

print(
    f"ONNX per sample    : "
    f"{(onnx_time / len(X)) * 1000:.4f} ms"
)


# ============================================================
# MODEL SIZE
# ============================================================

sklearn_size = os.path.getsize(MODEL_PATH)
onnx_size = os.path.getsize(ONNX_PATH)

print("\n" + "=" * 60)
print("MODEL SIZE")
print("=" * 60)

print(
    f"Sklearn model : "
    f"{sklearn_size / (1024 * 1024):.2f} MB"
)

print(
    f"ONNX model    : "
    f"{onnx_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

comparison = pd.DataFrame({
    "Actual": y[:10],
    "Sklearn": sklearn_predictions[:10],
    "ONNX": onnx_predictions[:10],
    "Difference": absolute_difference[:10]
})

print(comparison.to_string(index=False))


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

if np.allclose(
    sklearn_predictions,
    onnx_predictions,
    rtol=1e-4,
    atol=1e-5
):
    print("PASS: ONNX predictions match sklearn predictions.")
else:
    print("WARNING: ONNX predictions differ from sklearn predictions.")

print("=" * 60)