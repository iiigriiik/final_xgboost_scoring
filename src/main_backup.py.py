from sklearn.metrics import classification_report, roc_auc_score
import os
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import json
import datetime


# Список признако
FEATURES = [
    "risky_tx_count",
    "avg_risk_factor",
    "total_collateral_eth",
    "risk_factor_above_threshold_daily_count",
    "unique_lending_protocol_count",
    "incoming_tx_count",
    "borrow_amount_sum_eth",
    "repay_amount_sum_eth",
    "wallet_age",
    "total_gas_paid_eth",
]

TARGET = "target"


def load_data(path):
    """Загрузка данных из .parquet"""
    print(f"[INFO] Загрузка данных из {path}")
    return pd.read_parquet(path)


def analyze_data(df):
    """Анализ данных и построение графиков"""
    print("[INFO] Анализ данных...")
    print("Баланс классов:")
    print(df[TARGET].value_counts())

    # График баланса классов
    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET, data=df)
    plt.title("Распределение целевой переменной")
    plt.savefig("reports/target_distribution.png")
    plt.close()

    # Корреляция с целевой переменной
    plt.figure(figsize=(10, 8))
    df[FEATURES + [TARGET]].corr()[[TARGET]].sort_values(by=TARGET).plot(kind="barh")
    plt.title("Корреляция признаков с целевой переменной")
    plt.savefig("reports/features_correlation.png")
    plt.close()


def train_model(X_train, y_train, model_path="../models/best_xgboost_model.pkl"):
    """Обучение модели"""
    print("[INFO] Обучение модели...")
    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)

    print(f"[INFO] Сохранение модели в {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model


def evaluate_model(model, X_test, y_test):
    """Оценка модели"""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    roc_auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Сохраняем метрики в файл
    log_metrics(
        {
            "classification_report": classification_report(
                y_test, preds, output_dict=True
            ),
            "roc_auc": roc_auc,
        }
    )





def log_metrics(metrics, filename="reports/metrics_log.jsonl"):
    """Логирование метрик модели"""
    timestamp = datetime.datetime.now().isoformat()
    entry = {"timestamp": timestamp, **metrics}

    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[INFO] Метрики записаны в {filename}")


def evaluate_model_from_file(data_path, model_path="../models/best_xgboost_model.pkl"):
    """Оценка модели из файла без переобучения"""
    df = load_data(data_path)
    model = joblib.load(model_path)
    X = df[FEATURES]
    y_true = df[TARGET]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    roc_auc = roc_auc_score(y_true, probs)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Логируем метрики
    log_metrics(
        {
            "classification_report": classification_report(
                y_true, preds, output_dict=True
            ),
            "roc_auc": roc_auc,
        }
    )


def predict_new_data(data_path, model_path="../models/best_xgboost_model.pkl"):
    """Предсказание на новых данных"""
    df = load_data(data_path)
    model = joblib.load(model_path)
    X = df[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]
    df["probability"] = probabilities
    output_path = data_path.replace(".parquet", "_predicted.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Предсказания сохранены в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Команда: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Анализ данных")
    analyze_parser.add_argument(
    '--data_path', 
    type=str, 
    required=True, 
    help='Путь к dataset.parquet'
)

    # Команда: train
    train_parser = subparsers.add_parser("train", help="Обучение модели")
    train_parser.add_argument(
        "--data_path", type=str, required=True, help="Путь к dataset.parquet"
    )

    # Команда: predict
    predict_parser = subparsers.add_parser(
        "predict", help="Предсказание на новых данных"
    )
    predict_parser.add_argument(
        "--data_path", type=str, required=True, help="Путь к test_data.parquet"
    )

    # Команда: evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Оценка модели без переобучения"
    )
    evaluate_parser.add_argument(
        "--data_path", type=str, required=True, help="Путь к данным с target"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        df = load_data(args.data_path)
        analyze_data(df)

    elif args.command == "train":
        df = load_data(args.data_path)
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    elif args.command == "predict":
        predict_new_data(args.data_path)

    elif args.command == "evaluate":
        evaluate_model_from_file(args.data_path)
