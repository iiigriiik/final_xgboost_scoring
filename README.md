# Credit Scoring Model

Проект реализует скоринговую модель дефолта заёмщика на основе XGBoost.

API принимает JSON и возвращает вероятность дефолта.

---

## 🧩 Описание

Модель обучена на данных о транзакциях кошельков и определяет вероятность дефолта по 10 признакам.

Пример ответа:

```json
{
  "defaulter": true,
  "probability": 0.86
}

📁 Структура проекта
.
├── config/
│   └── config.yaml         ← признаки и TARGET
├── models/                  ← место для модели
│   └── best_xgboost_model.pkl
├── src/
│   ├── data_loader.py       ← загрузка данных
│   ├── model_trainer.py     ← обучение модели
│   ├── model_predictor.py   ← предсказание
│   ├── data_analyzer.py     ← анализ данных
│   ├── metrics_logger.py    ← логирование метрик
│   └── utils.py             ← утилиты и вспомогательные функции
├── api.py                   ← FastAPI сервер
├── requirements.txt         ← зависимости
├── Dockerfile               ← инструкции для контейнеризации
├── .gitignore               ← какие файлы игнорируем
└── README.md                ← этот файл
Установка зависимостей:
pip install -r requirements.txt
Запуск API:
set PYTHONPATH=D:\jupiter\ypiter\final_xgboost_scoring
uvicorn api:app --reload
http://127.0.0.1:8000/docs
Сборка образа:
docker build -t credit-scoring .
Запуск контейнера:
docker run --rm -p 8000:80 -v $(pwd)/models:/app/models credit-scoring
MIT

---

## ✅ Сохрани как `README.md`  
(не забудь тип файла → "Все файлы (*.*)")

📌 Этот `README.md`:
- Понятен другим разработчикам
- Объясняет, как работать с проектом локально и в Docker
- Не содержит лишних файлов
- Готов к GitHub

---

## 🗂 Шаг 2: Нужно ли добавлять `best_xgboost_model.pkl` в репозиторий?

### Вариант A: Хочу, чтобы проект работал сразу после клонирования  
→ Добавь модель в папку `models/` и закоммить её

```powershell
copy D:\путь_к_твоей_модели\best_xgboost_model.pkl models\