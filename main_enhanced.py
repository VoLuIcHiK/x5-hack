from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from functools import lru_cache
import json

# Настройка логирования для мониторинга
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем текущую директорию в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Импорт модели
try:
    from models.ner_model_enhanced import NERModel, EntityResult
    logger.info("✅ Successfully imported NERModel")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)

# Инициализация FastAPI с оптимизированными настройками
app = FastAPI(
    title="🚀 High-Performance NER Service",
    version="3.0.0",
    description="Optimized NER service for hackathon with <1s response time",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware для публичного доступа
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для оптимизации
MODEL_INSTANCE = None
EXECUTOR = ThreadPoolExecutor(max_workers=4)  # Пул потоков для CPU-интенсивных задач

def get_model_path():
    """Умный поиск пути к модели"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(current_dir, "model_files"),
        "./model_files",
        "/app/model_files",
        os.path.expanduser("~/model_files"),
        "model_files"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found model directory: {path}")
            return path
    
    logger.warning("❌ Model directory not found, using fallback")
    return possible_paths[0]

# Инициализация модели при старте приложения
@app.on_event("startup")
async def startup_event():
    """Инициализация ресурсов при запуске"""
    global MODEL_INSTANCE
    
    logger.info("🚀 Starting High-Performance NER Service...")
    start_time = time.time()
    
    MODEL_PATH = get_model_path()
    logger.info(f"📂 Model path: {MODEL_PATH}")
    
    try:
        MODEL_INSTANCE = NERModel(model_dir=MODEL_PATH)
        logger.info("✅ Model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing model: {e}")
        MODEL_INSTANCE = NERModel(model_dir=None)  # Fallback
    
    init_time = time.time() - start_time
    logger.info(f"⚡ Startup completed in {init_time:.2f}s")

# Pydantic модели
class PredictRequest(BaseModel):
    input: str
    
    class Config:
        schema_extra = {
            "example": {
                "input": "молоко 3.2% 1 литр Домик в деревне"
            }
        }

class PredictResponse(BaseModel):
    entities: List[EntityResult]
    processing_time: float
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    uptime: float
    model_loaded: bool
    response_time: float

# Кэширование для часто запрашиваемых результатов
@lru_cache(maxsize=1000)
def cached_predict(text: str):
    """Кэшированное предсказание для одинаковых запросов"""
    return MODEL_INSTANCE.extractor.predict(text)

# Основной API endpoint - ОПТИМИЗИРОВАННЫЙ ДЛЯ СКОРОСТИ
@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    🚀 ВЫСОКОПРОИЗВОДИТЕЛЬНЫЙ ENDPOINT для извлечения сущностей
    
    Оптимизирован для:
    - Время отклика < 1 секунды
    - Асинхронная обработка
    - Масштабируемость под нагрузкой
    """
    start_time = time.time()
    
    # Валидация входных данных
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    if len(request.input) > 1000:  # Ограничение длины для скорости
        raise HTTPException(status_code=400, detail="Input text too long (max 1000 characters)")
    
    try:
        # Асинхронное выполнение предсказания в пуле потоков
        loop = asyncio.get_event_loop()
        
        # Используем кэшированную версию для ускорения
        entities = await loop.run_in_executor(
            EXECUTOR,
            cached_predict,
            request.input.strip()
        )
        
        # Преобразуем в формат EntityResult
        results = []
        for entity in entities:
            results.append(EntityResult(
                start_index=entity['start_index'],
                end_index=entity['end_index'],
                entity=entity['entity'],
                text=entity.get('text', request.input[entity['start_index']:entity['end_index']])
            ))
        
        processing_time = time.time() - start_time
        
        # Логирование для мониторинга производительности
        logger.info(f"⚡ Processed request in {processing_time:.3f}s, found {len(results)} entities")
        
        # Предупреждение если время превышает требование
        if processing_time > 0.9:
            logger.warning(f"🐌 Slow response: {processing_time:.3f}s")
        
        return PredictResponse(
            entities=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error processing request in {processing_time:.3f}s: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# Облегченный health check для мониторинга
@app.get("/health", response_model=HealthResponse)
async def health():
    """Быстрая проверка статуса сервиса"""
    start_time = time.time()
    
    # Получаем время работы сервиса
    uptime = time.time() - startup_time if 'startup_time' in globals() else 0
    
    response_time = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if MODEL_INSTANCE else "degraded",
        uptime=uptime,
        model_loaded=MODEL_INSTANCE is not None,
        response_time=response_time
    )

# Простая главная страница для демонстрации
@app.get("/", response_class=HTMLResponse)
async def root():
    """Простая демо-страница"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 High-Performance NER Service</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .container { background: #f8f9fa; padding: 30px; border-radius: 10px; }
            h1 { color: #2563eb; text-align: center; }
            .demo { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
            code { background: #e5e7eb; padding: 2px 4px; border-radius: 3px; }
            .status { color: #059669; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 High-Performance NER Service</h1>
            <div class="status">✅ Service is running and ready for hackathon!</div>
            
            <div class="demo">
                <h3>📡 API Usage</h3>
                <p><strong>Endpoint:</strong> <code>POST /api/predict</code></p>
                <p><strong>Example:</strong></p>
                <pre><code>curl -X POST "/api/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"input": "молоко 3.2% 1 литр Домик в деревне"}'</code></pre>
            </div>
            
            <div class="demo">
                <h3>🎯 Performance Features</h3>
                <ul>
                    <li>⚡ Response time &lt; 1 second</li>
                    <li>🔄 Async processing</li>
                    <li>📈 Scalable under load</li>
                    <li>🌐 Public access ready</li>
                </ul>
            </div>
            
            <div class="demo">
                <h3>🔗 Useful Links</h3>
                <p><a href="/docs">📖 API Documentation</a></p>
                <p><a href="/health">❤️ Health Check</a></p>
            </div>
        </div>
    </body>
    </html>
    """

# Endpoint для получения метрик (для мониторинга на хакатоне)
@app.get("/metrics")
async def metrics():
    """Метрики для мониторинга производительности"""
    return {
        "service": "high-performance-ner",
        "version": "3.0.0",
        "model_loaded": MODEL_INSTANCE is not None,
        "cache_info": cached_predict.cache_info()._asdict(),
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0,
        "status": "production-ready"
    }

# Сохраняем время запуска для uptime
startup_time = time.time()

if __name__ == "__main__":
    print("🚀 Starting High-Performance NER Service for Hackathon!")
    print("="*60)
    print("🎯 Optimized for:")
    print("   ⚡ Response time < 1 second")
    print("   🔄 Async processing")
    print("   📈 High scalability")
    print("   🌐 Public deployment ready")
    print("="*60)
    
    # Оптимизированные настройки uvicorn для максимальной производительности
    uvicorn.run(
        app,
        host="0.0.0.0",  # Публичный доступ
        port=int(os.environ.get("PORT", 8000)),  # Поддержка переменной окружения
        workers=1,  # Один worker для начала (можно увеличить)
        loop="asyncio",  # Быстрый event loop
        http="httptools",  # Быстрый HTTP parser
        log_level="info",
        access_log=True
    )