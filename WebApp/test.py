import joblib

model = joblib.load("sleep_fatigue_model.pkl")

preprocessor = model.named_steps["preprocessor"]

print("\nNUMERICAL FEATURES")
print(preprocessor.transformers_[0][2])

print("\nCATEGORICAL FEATURES")
print(preprocessor.transformers_[1][2])

encoder = preprocessor.named_transformers_["cat"]

print("\nCATEGORIES")
for feature, categories in zip(
    preprocessor.transformers_[1][2],
    encoder.categories_
):
    print(f"\n{feature}")
    print(categories)