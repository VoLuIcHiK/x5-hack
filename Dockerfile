# Минимальный Dockerfile для хакатона
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Копируем весь проект
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
# Запуск
CMD ["python", "main_final.py"]