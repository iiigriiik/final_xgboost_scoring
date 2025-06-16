import json
import datetime


def log_metrics(metrics, filename="reports/metrics_log.jsonl"):
    """Логирование метрик модели"""
    timestamp = datetime.datetime.now().isoformat()
    entry = {"timestamp": timestamp, **metrics}

    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[INFO] Метрики записаны в {filename}")
