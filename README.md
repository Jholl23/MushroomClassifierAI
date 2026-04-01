# Mushroom Classifier AI

A machine learning project that predicts whether a mushroom is **edible** or **poisonous** based on its physical characteristics.

## Overview

This project uses the **Mushroom** dataset and a **Random Forest classifier** wrapped inside a scikit-learn pipeline.

The app allows the user to select mushroom features from a simple web interface and get:

- a prediction (`Edible` or `Poisonous`)
- class probabilities
- model accuracy
- the selected input values

## Project structure

```text
MushroomClassifierAI/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
├── data/
├── models/
│   ├── metrics.json
│   ├── mushroom_pipeline.joblib
│   └── schema.json
└── src/
    └── train.py
```

## Tech stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib
- ucimlrepo
- matplotlib

## Model

The training pipeline includes:

- categorical feature preprocessing with `OneHotEncoder`
- `RandomForestClassifier`
- train/test split with stratification

## Results

Current model performance:

- **Accuracy:** `1.0000`

## How to run

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

### 4. Run the app

```bash
streamlit run app.py
```

## Notes

This is an **educational project** for machine learning practice and portfolio building.

It must **not** be used to decide whether a real mushroom is safe to eat.

## Future improvements

- Add friendlier explanations for each feature
- Add feature importance visualization
- Improve UI styling
- Compare multiple models
