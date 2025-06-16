# Credit Scoring Model

Проект реализует скоринговую модель дефолта заёмщика на основе XGBoost

---

## 🧩 Описание

Модель обучена на данных о транзакциях кошельков и определяет вероятность дефолта заёмщика по 10 признакам.

API принимает JSON и возвращает:

```json
{
  "defaulter": true,
  "probability": 0.86
}

📁 Структура проекта

.
├── data/                     ← сырые данные (не обязательны)
│   └── .gitkeep              ← чтобы папка была в репозитории
├── models/                   ← модель (можно не выкладывать)
│   └── best_xgboost_model.pkl
├── reports/                  ← графики и метрики
│   └── .gitkeep
├── src/                      ← все модули Python
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── model_predictor.py
│   ├── data_analyzer.py
│   ├── metrics_logger.py
│   └── utils.py
├── api.py                    ← FastAPI сервер
├── requirements.txt          ← зависимости
├── .gitignore                ← игнорируем лишнее
└── README.md                 ← этот файл

Установка зависимостей:
pip install -r requirements.txt

Запуск API:
set PYTHONPATH=D:\jupiter\ypiter\final_xgboost_scoring
uvicorn api:app --reload
http://127.0.0.1:8000/docs

📦 Пример запроса к /predict
{
  "risky_tx_count": 5,
  "avg_risk_factor": 0.6,
  "total_collateral_eth": 1.2,
  "risk_factor_above_threshold_daily_count": 3,
  "unique_lending_protocol_count": 2,
  "incoming_tx_count": 10,
  "borrow_amount_sum_eth": 0.5,
  "repay_amount_sum_eth": 0.2,
  "wallet_age": 365,
  "total_gas_paid_eth": 0.1
}

Вернёт:
{
  "defaulter": true,
  "probability": 0.86
}

🛡 Лицензия
MIT

---

## 🧪 Шаг 3: Сохрани и проверь

1. Вставь этот текст в Блокнот
2. Сохрани как:

D:\jupiter\ypiter\final_xgboost_scoring\README.md


3. При сохранении:
   - Имя файла: `README.md`
   - Тип файла: **Все файлы (*.\*)**
   - Не добавляй `.txt` или `.md.md`!

---

## 📌 После этого выполни:

```powershell
dir
→ Должен(а) увидеть:
README.md