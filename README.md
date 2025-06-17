# Credit Scoring Model

Проект реализует скоринговую модель дефолта заёмщика на основе XGBoost. API принимает JSON и возвращает вероятность дефолта.

## Описание

Модель обучена на данных о транзакциях кошельков и определяет вероятность дефолта по 10 признакам.

Пример ответа:
{
  "defaulter": true,
  "probability": 0.86
}

## Структура проекта

.
├── config/
│   └── config.yaml         # Признаки и TARGET
├── src/
│   ├── data_loader.py       # Загрузка данных
│   ├── model_trainer.py     # Обучение модели
│   ├── model_predictor.py   # Предсказание
│   ├── data_analyzer.py     # Анализ данных
│   ├── metrics_logger.py    # Логирование метрик
│   └── utils.py             # Утилиты
├── api.py                   # FastAPI сервер
├── requirements.txt         # Зависимости
├── Dockerfile               # Инструкции для Docker
├── EDA.ipynb                # Исследовательский анализ данных
├── .gitignore               # Какие файлы игнорируем
└── README.md                # Этот файл

## Установка и запуск

Установка зависимостей:
pip install -r requirements.txt

Запуск API локально:
set PYTHONPATH=D:\jupiter\ypiter\final_xgboost_scoring
uvicorn api:app --reload

Открой в браузере:
http://127.0.0.1:8000/docs

## Запуск через Docker

Сборка образа:
docker build -t credit-scoring .

Запуск контейнера:
docker run --rm -p 8000:80 credit-scoring

## Лицензия: MIT