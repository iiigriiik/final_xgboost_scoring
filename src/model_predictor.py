import joblib
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from data_loader import load_data
from metrics_logger import log_metrics

# Загружаем конфиг
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

FEATURES = config["features"]
TARGET = config["target"]

MODEL_PATH = "models/xgboost_model.pkl"


def predict_new_data(data_path):
    """Предсказание на новых данных"""
    df = load_data(data_path)
    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]

    df['probability'] = probabilities
    output_path = data_path.replace(".parquet", "_predicted.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Предсказания сохранены в {output_path}")


def evaluate_model_from_file(data_path):
    """Оценка модели без переобучения"""
    df = load_data(data_path)
    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    y_true = df[TARGET]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    roc_auc = roc_auc_score(y_true, probs)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Логируем метрики
    log_metrics({
        "classification_report": classification_report(
            y_true, preds, output_dict=True
        ),
        "roc_auc": roc_auc
    })
