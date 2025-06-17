# Берём официальный образ Python
FROM python:3.9-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем проект
COPY твой проект в /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Команда запуска
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]