from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from metrics_logger import log_metrics


def train_model(
    X_train,
    y_train,
    model_path="../models/best_xgboost_model.pkl"
):
    """Обучение модели"""
    print("[INFO] Обучение модели...")
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    print(f"[INFO] Сохранение модели в {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model


def evaluate_model(
    model,
    X_test,
    y_test
):
    """Оценка модели"""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    roc_auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Логируем метрики
    log_metrics(
        {
            "classification_report": classification_report(
                y_test, preds, output_dict=True
            ),
            "roc_auc": roc_auc
        }
    )
