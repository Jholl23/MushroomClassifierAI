from pathlib import Path
import json
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # 1. Cargar el dataset Mushroom desde UCI
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features.copy()
    y = mushroom.data.targets.copy().squeeze()

    # 2. Convertir a texto por simplicidad
    X = X.astype(str)
    y = y.astype(str)

    # 3. Guardar esquema de opciones para usar luego en la app
    schema = {}
    for col in X.columns:
        values = sorted(X[col].dropna().astype(str).unique().tolist())
        schema[col] = values

    with open(MODELS_DIR / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # 4. Separar entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Preprocesado: OneHotEncoding para variables categóricas
    categorical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # 6. Modelo
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # 7. Pipeline completo
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # 8. Entrenar
    pipeline.fit(X_train, y_train)

    # 9. Evaluar
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 10. Guardar pipeline entrenado
    joblib.dump(pipeline, MODELS_DIR / "mushroom_pipeline.joblib")

    print("Modelo entrenado y guardado en models/")
    print(f"Accuracy: {acc:.4f}")
    print("Clases:", sorted(y.unique().tolist()))
    print("Archivos generados:")
    print("- models/mushroom_pipeline.joblib")
    print("- models/schema.json")
    print("- models/metrics.json")


if __name__ == "__main__":
    main()