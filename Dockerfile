# Берём официальный образ Python
FROM python:3.9-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем config.yaml и requirements.txt
COPY config/config.yaml .
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY src/ src/
COPY api.py .

# Команда запуска
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]