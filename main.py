import os
import sys
import time
import logging
import asyncio
from typing import List, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Добавляем текущую директорию в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Импорт модели
try:
    from models.ner_model_enhanced import NERModel
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Минимальная настройка логирования для максимальной производительности
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("uvicorn.error")

# Pydantic модели - точно по спецификации API
class PredictRequest(BaseModel):
    input: str

class EntityResult(BaseModel):
    start_index: int
    end_index: int
    entity: str

# FastAPI приложение с минимальными настройками для скорости
app = FastAPI(
    title="High-Performance NER Service",
    version="5.0.0",
    description="Ultra-fast NER for hackathon",
    docs_url=None,  # Отключаем docs для скорости
    redoc_url=None  # Отключаем redoc для скорости
)

# Минимальные CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Глобальные переменные для оптимизации
MODEL_INSTANCE = None
EXECUTOR = ThreadPoolExecutor(max_workers=4)  # Оптимизировано для CPU
startup_time = time.time()

def get_model_path():
    """Быстрый поиск пути к модели"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_files")
    return model_path

# Инициализация модели при старте
@app.on_event("startup")
async def startup_event():
    global MODEL_INSTANCE, startup_time
    startup_time = time.time()

    model_dir = get_model_path()
    try:
        MODEL_INSTANCE = NERModel(model_dir=model_dir)
        # Прогрев модели для ускорения первых запросов
        _ = MODEL_INSTANCE.extractor.predict("тест")
    except Exception as e:
        print(f"Model init error: {e}", file=sys.stderr)
        MODEL_INSTANCE = NERModel(model_dir=None)

# Высокопроизводительное кэширование
@lru_cache(maxsize=10000)
def cached_predict(text: str):
    """Кэшированное предсказание с большим кэшем"""
    if not MODEL_INSTANCE:
        return []
    return MODEL_INSTANCE.extractor.predict(text)

# Основной endpoint точно по спецификации API
@app.post("/api/predict", response_model=List[EntityResult])
async def predict(request: PredictRequest):
    # Обработка пустого input - возвращаем пустой список
    if not request.input or not request.input.strip():
        return []

    # Ограничение длины для производительности
    if len(request.input) > 1000:
        raise HTTPException(status_code=400, detail="Input too long")

    try:
        # Асинхронная обработка в отдельном потоке
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            EXECUTOR,
            cached_predict,
            request.input.strip()
        )

        # Преобразование в нужный формат ответа
        return [EntityResult(**entity) for entity in entities]

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

# Минимальный health check
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL_INSTANCE is not None}

# Оптимизированный запуск
if __name__ == "__main__":
    # Максимальная производительность для single worker deployment
    uvicorn.run(
        "main_final:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=1,  # Single worker для хакатона
        loop="asyncio",
        http="httptools",
        log_level="error",  # Минимальное логирование
        access_log=False,  # Без access логов для скорости
        reload=False
    )
