import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType


# Load the trained sklearn pipeline
model = joblib.load("sleep_fatigue_model.pkl")


# Define the exact input schema
initial_types = [
    ("age", FloatTensorType([None, 1])),
    ("bedtime_phone_minutes", FloatTensorType([None, 1])),
    ("screen_brightness_pct", FloatTensorType([None, 1])),
    ("caffeine_post_5pm_mg", FloatTensorType([None, 1])),
    ("physical_activity_min", FloatTensorType([None, 1])),
    ("morning_alarm_snoozes", FloatTensorType([None, 1])),

    ("gender", StringTensorType([None, 1])),
    ("occupation_type", StringTensorType([None, 1])),
    ("chronotype", StringTensorType([None, 1])),
    ("primary_bedtime_app", StringTensorType([None, 1])),

    ("blue_light_filter_active", Int64TensorType([None, 1])),
]


# Convert sklearn Pipeline -> ONNX
onnx_model = convert_sklearn(
    model,
    initial_types=initial_types,
    target_opset=17,
)


# Save ONNX model
with open("sleep_fatigue.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


print("ONNX conversion successful!")
print("Saved: sleep_fatigue.onnx")