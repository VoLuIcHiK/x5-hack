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


class SoftValidationParser:
    def __init__(self):
        # единицы и ключевые слова для процентов остались прежними
        # шаблоны единиц
        self.volume_units = [
            'г', 'гр\\.?', 'грамм\\.?',
            'кг', 'килограмм\\.?',
            'мл', 'мл\\.?', 'миллилитр\\.?',
            'л', 'л\\.?', 'литр\\.?',
            'шт\\.?', 'штук\\.?'
        ]
        self.percent_units = ['%', 'процент(ов)?', 'проц\\.?', 'жирн\\.?']

        # комбинируем единицы в один шаблон
        self.vol_pattern = re.compile(
            rf'(\d+[.,]?\d*)\s*({"|".join(self.volume_units)})',
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
    
    
# Исправленная версия экстрактора с решением всех трех проблем
class ImprovedProductEntityExtractor:
    def __init__(self,
                 ner_model_path="./ner_ecommerce_model",
                 embedding_model_name="cointegrated/rubert-tiny2", #'DeepPavlov/rubert-base-cased'
                 catalog_brands=None,
                 catalog_categories=None):

        self.tokenizer_ner = AutoTokenizer.from_pretrained(ner_model_path,
                                                           is_split_into_words=True)
        self.model_ner = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        # ipeline NER с правильной агрегацией - ПРОСТАЯ СТРАТЕГИЯ
        self.pipeline_ner = pipeline(
            "ner",
            model=self.model_ner,
            tokenizer=self.tokenizer_ner,
            aggregation_strategy="simple",  # Возвращаем к simple!
            device=-1
        )

        # Инициализация остальных компонентов
        self.embedding_model = SentenceTransformer(embedding_model_name)
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

    def _postprocess_type_brand(self, annotations, text):
        """Исправленная постобработка TYPE/BRAND с правильными BIO-метками"""
        tokens = text.split()

        # Если два span подряд покрывают одно слово – оба меняем на TYPE
        if (len(annotations) == 2 and
            annotations[0][2] == 'B-TYPE' and annotations[1][2] == 'B-BRAND' and
            annotations[0][1] == annotations[1][0]):
            start, _, _ = annotations[0]
            _, end, _ = annotations[1]
            return [(start, end, 'B-TYPE')]

        # Если только первое слово размечено как B-TYPE, а слов больше — отмечаем все как TYPE
        if (len(annotations) == 1 and
            annotations[0][2] == 'B-TYPE' and
            len(tokens) > 1):

            spans = []
            char_pos = 0

            for i, token in enumerate(tokens):
                token_start = char_pos
                token_end = char_pos + len(token)

                # Первый токен - B-TYPE, остальные - I-TYPE
                if i == 0:
                    spans.append((token_start, token_end, 'B-TYPE'))
                else:
                    spans.append((token_start, token_end, 'I-TYPE'))

                # Переходим к следующему токену (учитываем пробел)
                char_pos = token_end + 1

            return spans

        # Если аннотация покрывает только часть первого слова, расширяем на все слова
        if (len(annotations) == 1 and
            annotations[0][2] == 'B-TYPE' and
            len(tokens) > 0):

            # Проверяем, покрывает ли аннотация весь первый токен
            first_token_end = len(tokens[0])
            if annotations[0][1] < first_token_end and len(tokens) > 1:
                spans = []
                char_pos = 0

                for i, token in enumerate(tokens):
                    token_start = char_pos
                    token_end = char_pos + len(token)

                    if i == 0:
                        spans.append((token_start, token_end, 'B-TYPE'))
                    else:
                        spans.append((token_start, token_end, 'I-TYPE'))

                    char_pos = token_end + 1

                return spans

        return annotations

    def _map_entities_to_original_words(self, ner_results, original, corrected):
        """
        КЛЮЧЕВАЯ ФУНКЦИЯ: Правильное сопоставление NER результатов с исходными словами
        """
        if not ner_results:
            return []

        original_tokens = original.split()
        corrected_tokens = corrected.split()

        # Создаем карту: позиция в corrected -> позиция в original
        entities = []

        for ner_ent in ner_results:
            start_corr = ner_ent['start']
            end_corr = ner_ent['end']
            entity_text = corrected[start_corr:end_corr]
            label = f"B-{ner_ent['entity_group'].upper()}"

            # Находим соответствующие токены в corrected
            char_pos = 0
            found_token_indices = []

            for i, token in enumerate(corrected_tokens):
                token_start = char_pos
                token_end = char_pos + len(token)

                # Если токен пересекается с entity
                if start_corr < token_end and end_corr > token_start:
                    found_token_indices.append(i)

                char_pos = token_end + 1  # +1 для пробела

            # Мапим найденные токены на исходный текст
            if found_token_indices:
                # Берем границы от первого до последнего найденного токена
                first_token_idx = found_token_indices[0]
                last_token_idx = found_token_indices[-1]

                # Вычисляем позиции в исходном тексте
                orig_start = sum(len(original_tokens[i]) + 1 for i in range(first_token_idx))
                orig_end = orig_start + sum(len(original_tokens[i]) + 1 for i in range(first_token_idx, last_token_idx + 1)) - 1

                # Корректируем границы
                if orig_start < len(original):
                    if orig_end > len(original):
                        orig_end = len(original)
                    entities.append((orig_start, orig_end, label))

        return entities

    def extract_entities_with_soft_validation(self, query: str):
        """
        Унифицированный метод, который сначала лемматизирует текст,
        затем вызывает predict() для получения BIO-разметки на уровне слов,
        и дополнительно возвращает raw-список для дальнейшей обработки.
        """
        # 1. Spell correction / лемматизация
        tokens = query.split()
        corrected = " ".join(self.lemmatize(tok) for tok in tokens)

        # 2. Получаем BIO-аннотации на уровне слов
        word_entities = self.predict(query)

        # 3. Собираем полный результат
        return {
            "original_query": query,
            "corrected_query": corrected,
            "annotations": word_entities,
            "type":   self._get_entity_text_list(word_entities, query, 'TYPE'),
            "brand":  self._get_entity_text_list(word_entities, query, 'BRAND'),
            "volumes":  [e for e in word_entities if e['entity'].endswith('-VOLUME')],
            "percents": [e for e in word_entities if e['entity'].endswith('-PERCENT')]
        }

    def _get_entity_text_list(self, entities, text, entity_type):
        """Возвращает список вхождений entity_type из annotations."""
        values = []
        for ent in entities:
            if ent['entity'].endswith(f"-{entity_type}"):
                values.append(text[ent['start_index']:ent['end_index']])
        return values

    def extract_entities_with_soft_validation(self, query: str):
        """
        Унифицированный метод, который сначала лемматизирует текст,
        затем вызывает predict() для получения BIO-разметки на уровне слов,
        и дополнительно возвращает raw-список для дальнейшей обработки.
        """
        # 1. Spell correction / лемматизация
        tokens = query.split()
        corrected = " ".join(self.lemmatize(tok) for tok in tokens)

        # 2. Получаем BIO-аннотации на уровне слов
        word_entities = self.predict(query)

        # 3. Собираем полный результат
        return {
            "original_query": query,
            "corrected_query": corrected,
            "annotations": word_entities,
            "type":   self._get_entity_text_list(word_entities, query, 'TYPE'),
            "brand":  self._get_entity_text_list(word_entities, query, 'BRAND'),
            "volumes":  [e for e in word_entities if e['entity'].endswith('-VOLUME')],
            "percents": [e for e in word_entities if e['entity'].endswith('-PERCENT')]
        }

    def predict(self, text: str):
        """
        Для каждого слова возвращает BIO-метки:
        – если слово попало под NER-модель или soft-parser, используем эти метки;
        – иначе TYPE.
        """
        # 1. Задаём границы слов
        words, spans = [], []
        pos = 0
        for w in text.split():
            words.append(w)
            spans.append((pos, pos + len(w)))
            pos += len(w) + 1

        # 2. Извлекаем NER spans
        ner_spans = [
            (r["start"], r["end"], r["entity_group"].upper())
            for r in self.pipeline_ner(text, aggregation_strategy="simple")
        ]
        # 3. Извлекаем volume/percent spans
        soft_spans = self.soft_parser.parse_with_soft_validation(text)

        # 4. Для каждого слова выбираем приоритетную метку
        entities = []
        prev = None
        '''print('Spans: ', spans)
        print('Words: ', words)
        print('Ner_spans: ', ner_spans)
        print('Soft_spans: ', soft_spans)'''
        for (s, e), w in zip(spans, words):
            lbl = None
            # ищем NER
            for a, b, t in ner_spans:
                if s >= a and e <= b:
                    lbl = t
                    break
            # если нет NER, ищем volume/percent
            if lbl is None:
                for a, b, t in soft_spans:
                    if s >= a and e <= b:
                        lbl = t.split('-')[-1]
                        break
            # по умолчанию TYPE
            if lbl is None:
                lbl = "TYPE"
            # назначаем B- либо I-
            prefix = "B" if lbl != prev else "I"
            entities.append({
                "start_index": s,
                "end_index":   e,
                "entity":      f"{prefix}-{lbl}"
            })
            prev = lbl

        return entities

    def _remove_overlaps(self, entities):
        """Удаляет перекрывающиеся сущности, оставляя более точные"""
        if not entities:
            return []

        sorted_entities = sorted(entities, key=lambda x: (x[0], x[1] - x[0]))
        result = []

        for current in sorted_entities:
            overlapped = False
            for i, existing in enumerate(result):
                if (current[0] < existing[1] and current[1] > existing[0]):  # есть пересечение
                    # Оставляем более короткую (более точную) сущность
                    if (current[1] - current[0]) <= (existing[1] - existing[0]):
                        result[i] = current
                    overlapped = True
                    break

            if not overlapped:
                result.append(current)

        return result

    def _get_entity_text(self, entities, text, entity_type):
        """Извлекает текст сущности определенного типа"""
        for start, end, label in entities:
            if entity_type in label:
                return text[start:end]
        return None

    def _format_volume(self, entity, text):
        """Форматирует сущность объема"""
        start, end, label = entity
        return {
            'value': text[start:end],
            'original': text[start:end],
            'position': (start, end)
        }

    def _format_percent(self, entity, text):
        """Форматирует сущность процента"""
        start, end, label = entity
        return {
            'value': text[start:end],
            'original': text[start:end],
            'position': (start, end)
        }

    # Алиас для обратной совместимости
    def extract_entities(self, query):
        return self.extract_entities_with_soft_validation(query)