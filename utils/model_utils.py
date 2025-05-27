import tensorflow as tf
import numpy as np
import pandas as pd

# Load and preprocess Excel
excel_path = BASE_DIR / "data" / "profiles_data.xlsx"
df = pd.read_excel(excel_path)
df["Section Name"] = df["Section Name"].astype(str).str.strip()
class_names = df["Section Name"].dropna().unique().tolist()
class_names.sort()

# Load trained CNN model
model = tf.keras.models.load_model("model/cnn_shape_model.h5")

def predict_profile(image):
    img = image.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 128, 128, 1)
    prediction = model.predict(img)
    pred_idx = np.argmax(prediction)
    return class_names[pred_idx], prediction[0][pred_idx]

def get_profile_data(profile_name):
    match = df[df["Section Name"] == profile_name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()

try:
    df = pd.read_excel(excel_path)
except Exception as e:
    raise FileNotFoundError(f"Could not load Excel at {excel_path}: {e}")

