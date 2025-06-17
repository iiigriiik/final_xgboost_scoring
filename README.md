# Credit Scoring Model

Проект реализует скоринговую модель дефолта заёмщика на основе XGBoost.  
API принимает JSON и возвращает вероятность дефолта.

## Описание

Модель обучена на данных о транзакциях кошельков и определяет вероятность дефолта по 10 признакам.

Пример ответа:
{
  "defaulter": true,
  "probability": 0.86
}

## Установка и запуск

### Установка зависимостей:
pip install -r requirements.txt

### Запуск API локально:
set PYTHONPATH=D:\jupiter\ypiter\final_xgboost_scoring
uvicorn api:app --reload

Открой в браузере:
http://127.0.0.1:8000/docs

## Запуск через Docker

### Сборка образа:
docker build -t credit-scoring .

### Запуск контейнера:
docker run --rm -p 8000:80 credit-scoring

## Лицензия: MIT