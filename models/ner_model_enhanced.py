import numpy as np
import pandas as pd
import random
import re
import json
import asyncio
import pymorphy3
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, pipeline
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast, pipeline
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from ast import literal_eval
import evaluate
from functools import partial
from pydantic import BaseModel
from typing import List
import os

# Класс для результатов FastAPI
class EntityResult(BaseModel):
    start_index: int
    end_index: int
    entity: str
    text: str = ""

class SoftValidationParser:
    def __init__(self):
        # шаблоны единиц
        self.volume_units = [
            'г', r'гр\.?', r'грамм\.?',
            'кг', r'килограмм\.?',
            'мл', r'мл\.?', r'миллилитр\.?',
            'л', r'л\.?', r'литр\.?',
            r'шт\.?', r'штук\.?'
        ]
        
        self.percent_units = ['%', r'процент(?:ов)?', r'проц\.?', r'жирн\.?']
        
        # комбинируем единицы в один шаблон
        self.vol_pattern = re.compile(f'(\d+[.,]?\d*)\s*({"|".join(self.volume_units)})',
            flags=re.IGNORECASE
        )
        
        self.pct_pattern = re.compile(
            rf'(\d+[.,]?\d*)\s*({"|".join(self.percent_units)})',
            flags=re.IGNORECASE
        )

    def preprocess_volume_text(self, text):
        """Предварительная нормализация текста"""
        replacements = {
            # Нормализуем пробелы
            r'(\d+(?:[.,]\d+)?)\s*([а-яА-Я%]+)': r'\1 \2',
            r'(\d+(?:[.,]\d+)?)\s{2,}([а-яА-Я%]+)': r'\1 \2',
            # Стандартизируем десятичные разделители
            r'(\d+),(\d+)': r'\1.\2',
            # Единицы измерения
            r'\b(грамм?|гр\.?)\b': 'г',
            r'\b(килограмм?|кило|кг\.?)\b': 'кг',
            r'\b(миллилитр?|мл\.?)\b': 'мл',
            r'\b(литр?|л\.?)\b': 'л',
            r'\b(штук[аи]?|шт\.?)\b': 'шт',
            # Текстовые числа
            r'\bпол\b': '0.5',
            r'\bполтора\b': '1.5',
            r'\bполовин[ау]\b': '0.5',
            r'\bчетверть\b': '0.25',
            # Проценты и крепость
            r'\b(\d+(?:[.,]\d+)?)\s*%\b': r'\1 %',
            r'\b(\d+(?:[.,]\d+)?)\s*(процент(?:ов)?|проц\.?|жирн\.?|крепост(?:ь)?|градус(?:ов)?)\b': r'\1 %',
        }
        
        result = text
        for pattern, repl in replacements.items():
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
        return result

    def parse_with_soft_validation(self, text, type_entity=None):
        # 1. Препроцессинг
        normalized = self.preprocess_volume_text(text)
        entities = []

        # 2. Поиск объёмов
        for m in self.vol_pattern.finditer(normalized):
            start, end = m.span()
            span = normalized[start:end].split()
            offset = start
            for i, part in enumerate(span):
                p_start = normalized.find(part, offset)
                p_end = p_start + len(part)
                tag = 'B-VOLUME' if i == 0 else 'I-VOLUME'
                entities.append((p_start, p_end, tag))
                offset = p_end

        # 3. Поиск процентов
        for m in self.pct_pattern.finditer(normalized):
            start, end = m.span()
            span = normalized[start:end].split()
            offset = start
            for i, part in enumerate(span):
                p_start = normalized.find(part, offset)
                p_end = p_start + len(part)
                tag = 'B-PERCENT' if i == 0 else 'I-PERCENT'
                entities.append((p_start, p_end, tag))
                offset = p_end

        entities.sort(key=lambda x: x[0])
        return entities

# Основной экстрактор
class ImprovedProductEntityExtractor:
    def __init__(self,
                 ner_model_path="./model_files",
                 embedding_model_name="cointegrated/rubert-tiny2",
                 catalog_brands=None,
                 catalog_categories=None):
        
        # Загрузка NER модели с обработкой ошибок
        try:
            if ner_model_path and os.path.exists(ner_model_path):
                print(f"Loading custom NER model from: {ner_model_path}")
                self.tokenizer_ner = AutoTokenizer.from_pretrained(ner_model_path, is_split_into_words=True)
                self.model_ner = AutoModelForTokenClassification.from_pretrained(ner_model_path)
                
                # Pipeline NER с правильной агрегацией
                self.pipeline_ner = pipeline(
                    "ner",
                    model=self.model_ner,
                    tokenizer=self.tokenizer_ner,
                    aggregation_strategy="simple",
                    device=-1
                )
                print("✅ Custom NER model loaded successfully!")
            else:
                print(f"❌ NER model path not found: {ner_model_path}")
                self.tokenizer_ner = None
                self.model_ner = None
                self.pipeline_ner = None
                
        except Exception as e:
            print(f"❌ Error loading NER model from {ner_model_path}: {e}")
            self.tokenizer_ner = None
            self.model_ner = None
            self.pipeline_ner = None

        # Инициализация embedding модели
        try:
            if embedding_model_name:
                print(f"Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                print("✅ Embedding model loaded successfully!")
            else:
                self.embedding_model = None
        except Exception as e:
            print(f"❌ Warning: Could not load embedding model {embedding_model_name}: {e}")
            self.embedding_model = None

        self.catalog_brands = catalog_brands or []
        self.catalog_categories = catalog_categories or []
        self.morph = pymorphy3.MorphAnalyzer()
        
        self.id2label = {
            0: 'O',
            1: 'B-TYPE',
            2: 'I-TYPE',
            3: 'B-BRAND',
            4: 'I-BRAND',
            5: 'B-VOLUME',
            6: 'I-VOLUME',
            7: 'B-PERCENT',
            8: 'I-PERCENT'
        }
        
        # Парсер объемов
        self.soft_parser = SoftValidationParser()

    def lemmatize(self, word):
        """Лемматизация через pymorphy3"""
        if not word or not isinstance(word, str):
            return word or ""
        try:
            parsed = self.morph.parse(word)
            return parsed[0].normal_form if parsed else word.lower()
        except:
            return word.lower()

    def predict(self, text: str):
        """
        Основной метод предсказания, возвращающий BIO-метки для каждого слова.
        """
        # 1. Разбиваем на слова и считаем их spans
        words = text.split()
        spans = []
        pos = 0
        for w in words:
            spans.append((pos, pos + len(w)))
            pos += len(w) + 1

        # 2. Получаем сырые результаты
        ner_results = []
        if self.pipeline_ner:
            try:
                ner_results = list(self.pipeline_ner(text, aggregation_strategy="simple"))
            except Exception as e:
                print(f"Error in NER pipeline: {e}")
                
        soft_results = self.soft_parser.parse_with_soft_validation(text)

        # 3. Инициализируем группы для каждого слова
        word_groups = [None] * len(spans)

        # 4. Применяем soft-parser (они уже содержат B-/I- метки, берём просто группу)
        for start, end, label in soft_results:
            grp = label.split('-', 1)[1]
            for idx, (w_start, w_end) in enumerate(spans):
                if start < w_end and end > w_start:
                    word_groups[idx] = grp

        # 5. Применяем NER туда, где ещё нет soft-метки
        for r in ner_results:
            grp = r["entity_group"].upper()
            for idx, (w_start, w_end) in enumerate(spans):
                if word_groups[idx] is None and r["start"] < w_end and r["end"] > w_start:
                    word_groups[idx] = grp

        # 6. Строим BIO-метки по последовательности групп
        bio_tags = []
        prev_group = None
        for grp in word_groups:
            if grp is None:
                bio_tags.append('O')
                prev_group = None
            else:
                if grp == prev_group:
                    bio_tags.append(f'I-{grp}')
                else:
                    bio_tags.append(f'B-{grp}')
                prev_group = grp

        # 7. Возвращаем результат в формате, совместимом с FastAPI
        entities = []
        for i, (start, end) in enumerate(spans):
            if bio_tags[i] != 'O':
                entities.append({
                    'start_index': start,
                    'end_index': end,
                    'entity': bio_tags[i],
                    'text': text[start:end]
                })
        
        return entities

    def extract_entities_with_soft_validation(self, query):
        """Расширенный анализ с группировкой сущностей"""
        entities = self.predict(query)
        
        # Группируем сущности по типам
        type_entities = []
        brand_entities = []
        volume_entities = []
        percent_entities = []
        
        for entity in entities:
            entity_type = entity['entity']
            if 'TYPE' in entity_type:
                type_entities.append(entity)
            elif 'BRAND' in entity_type:
                brand_entities.append(entity)
            elif 'VOLUME' in entity_type:
                volume_entities.append(entity)
            elif 'PERCENT' in entity_type:
                percent_entities.append(entity)
        
        return {
            "original_query": query,
            "corrected_query": query,  # В будущем можно добавить коррекцию
            "annotations": [(e['start_index'], e['end_index'], e['entity']) for e in entities],
            "type": type_entities,
            "brand": brand_entities,
            "volumes": volume_entities,
            "percents": percent_entities
        }

# Основной класс NERModel для интеграции с FastAPI
class NERModel:
    def __init__(self, model_dir: str = "./model_files"):
        """
        Инициализация NER модели
        Args:
            model_dir: Путь к директории с моделью
        """
        self.model_dir = model_dir
        print(f"🚀 Initializing Enhanced NER Service...")
        print(f"📂 Model directory: {model_dir}")
        
        # Проверяем существование директории и файлов
        if os.path.exists(model_dir):
            print(f"✅ Model directory found: {model_dir}")
        else:
            print(f"❌ Model directory not found: {model_dir}")
        
        # Инициализируем экстрактор
        try:
            self.extractor = ImprovedProductEntityExtractor(
                ner_model_path=model_dir,
                embedding_model_name="cointegrated/rubert-tiny2"
            )
            print("🎉 Enhanced NER Model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing extractor: {e}")
            # Fallback без обученной модели
            self.extractor = ImprovedProductEntityExtractor(
                ner_model_path=None,
                embedding_model_name="cointegrated/rubert-tiny2"
            )

    async def predict(self, text: str) -> List[EntityResult]:
        """
        Асинхронное предсказание
        Args:
            text: Входной текст для анализа
        Returns:
            Список найденных сущностей в формате EntityResult
        """
        # Выполняем предсказание в отдельном потоке
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(None, self._sync_predict, text)
        
        # Преобразуем в формат EntityResult
        results = []
        for entity in entities:
            results.append(EntityResult(
                start_index=entity['start_index'],
                end_index=entity['end_index'],
                entity=entity['entity'],
                text=entity.get('text', text[entity['start_index']:entity['end_index']])
            ))
        
        return results

    def _sync_predict(self, text: str):
        """Синхронное предсказание для выполнения в executor"""
        return self.extractor.predict(text)

    def get_detailed_analysis(self, text: str):
        """Получить детальный анализ с дополнительной информацией"""
        return self.extractor.extract_entities_with_soft_validation(text)

    def info(self):
        """Расширенная информация о загруженной модели"""
        model_exists = os.path.exists(self.model_dir) if self.model_dir else False
        has_custom_model = self.extractor.pipeline_ner is not None
        has_embedding_model = self.extractor.embedding_model is not None
        
        return {
            "model_dir": self.model_dir,
            "status": "loaded",
            "custom_model_loaded": has_custom_model,
            "embedding_model_loaded": has_embedding_model,
            "model_directory_exists": model_exists,
            "entities": ["TYPE", "BRAND", "VOLUME", "PERCENT"],
            "features": [
                "custom_ner_model",
                "soft_validation_parser",
                "embedding_similarity",
                "postprocessing",
                "lemmatization"
            ],
            "fallback_methods": ["soft_parser", "rule_based"]
        }