from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "mushroom_pipeline.joblib"
SCHEMA_PATH = MODELS_DIR / "schema.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

FEATURE_LABELS = {
    "cap-shape": "Cap shape",
    "cap-surface": "Cap surface",
    "cap-color": "Cap color",
    "bruises": "Bruises",
    "odor": "Odor",
    "gill-attachment": "Gill attachment",
    "gill-spacing": "Gill spacing",
    "gill-size": "Gill size",
    "gill-color": "Gill color",
    "stalk-shape": "Stalk shape",
    "stalk-root": "Stalk root",
    "stalk-surface-above-ring": "Stalk surface above ring",
    "stalk-surface-below-ring": "Stalk surface below ring",
    "stalk-color-above-ring": "Stalk color above ring",
    "stalk-color-below-ring": "Stalk color below ring",
    "veil-type": "Veil type",
    "veil-color": "Veil color",
    "ring-number": "Ring number",
    "ring-type": "Ring type",
    "spore-print-color": "Spore print color",
    "population": "Population",
    "habitat": "Habitat",
}

VALUE_LABELS = {
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken",
        "x": "convex",
    },
    "cap-surface": {
        "f": "fibrous",
        "g": "grooves",
        "s": "smooth",
        "y": "scaly",
    },
    "cap-color": {
        "b": "buff",
        "c": "cinnamon",
        "e": "red",
        "g": "gray",
        "n": "brown",
        "p": "pink",
        "r": "green",
        "u": "purple",
        "w": "white",
        "y": "yellow",
    },
    "bruises": {
        "f": "no bruises",
        "t": "bruises",
    },
    "odor": {
        "a": "almond",
        "c": "creosote",
        "f": "foul",
        "l": "anise",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy",
        "y": "fishy",
    },
    "gill-attachment": {
        "a": "attached",
        "f": "free",
    },
    "gill-spacing": {
        "c": "close",
        "w": "crowded",
    },
    "gill-size": {
        "b": "broad",
        "n": "narrow",
    },
    "gill-color": {
        "b": "buff",
        "e": "red",
        "g": "gray",
        "h": "chocolate",
        "k": "black",
        "n": "brown",
        "o": "orange",
        "p": "pink",
        "r": "green",
        "u": "purple",
        "w": "white",
        "y": "yellow",
    },
    "stalk-shape": {
        "e": "enlarging",
        "t": "tapering",
    },
    "stalk-root": {
        "b": "bulbous",
        "c": "club",
        "e": "equal",
        "r": "rooted",
    },
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "k": "silky",
        "s": "smooth",
        "y": "scaly",
    },
    "stalk-surface-below-ring": {
        "f": "fibrous",
        "k": "silky",
        "s": "smooth",
        "y": "scaly",
    },
    "stalk-color-above-ring": {
        "b": "buff",
        "c": "cinnamon",
        "e": "red",
        "g": "gray",
        "n": "brown",
        "o": "orange",
        "p": "pink",
        "w": "white",
        "y": "yellow",
    },
    "stalk-color-below-ring": {
        "b": "buff",
        "c": "cinnamon",
        "e": "red",
        "g": "gray",
        "n": "brown",
        "o": "orange",
        "p": "pink",
        "w": "white",
        "y": "yellow",
    },
    "veil-type": {
        "p": "partial",
    },
    "veil-color": {
        "n": "brown",
        "o": "orange",
        "w": "white",
        "y": "yellow",
    },
    "ring-number": {
        "n": "none",
        "o": "one",
        "t": "two",
    },
    "ring-type": {
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
    },
    "spore-print-color": {
        "b": "buff",
        "h": "chocolate",
        "k": "black",
        "n": "brown",
        "o": "orange",
        "r": "green",
        "u": "purple",
        "w": "white",
        "y": "yellow",
    },
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary",
    },
    "habitat": {
        "d": "woods",
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
    },
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def format_option(feature: str, code: str) -> str:
    label = VALUE_LABELS.get(feature, {}).get(code, "unknown")
    return f"{code} — {label}"


st.set_page_config(
    page_title="Mushroom Classifier AI",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Classifier AI")
st.write("Predict whether a mushroom is edible or poisonous based on its physical characteristics.")

model = load_model()
schema = load_schema()
metrics = load_metrics()

with st.expander("Model metrics"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.write("Confusion matrix")
        st.write(metrics["confusion_matrix"])

st.subheader("Select mushroom characteristics")

columns = st.columns(2)
user_input = {}

for i, (feature, options) in enumerate(schema.items()):
    with columns[i % 2]:
        user_input[feature] = st.selectbox(
            format_feature_name(feature),
            options,
            format_func=lambda x, feature=feature: format_option(feature, x)
        )

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    label_map = {
        "e": "Edible",
        "p": "Poisonous"
    }

    predicted_label = label_map.get(prediction, prediction)
    prob_dict = {
        label_map.get(cls, cls): round(float(prob), 4)
        for cls, prob in zip(classes, probabilities)
    }

    st.subheader("Prediction result")

    if prediction == "e":
        st.success(f"Prediction: {predicted_label}")
    else:
        st.error(f"Prediction: {predicted_label}")

    st.write("Class probabilities")
    st.json(prob_dict)

    st.write("Selected values")
    st.dataframe(input_df, use_container_width=True)

st.caption("Educational project based on the UCI Mushroom dataset. Do not use this app to decide whether a real mushroom is safe to eat.")