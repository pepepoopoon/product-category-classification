# Классификация товарных категорий

## Описание задачи

Многоклассовая классификация товарного предложения по заголовку и описанию. Один
продавец не может одновременно присутствовать в обучении и контроле.

## Цель проекта

Показать воспроизводимый baseline классического NLP с честной оценкой переноса на
новых продавцов.

## Архитектура решения

CSV проходит строгую проверку схемы, затем продавцы детерминированно распределяются
между train/validation/test. Guard нормализованного `title + description` отклоняет
одинаковый текст товара, оказавшийся у разных продавцов в разных split. `FeatureUnion`
объединяет word TF-IDF (1–2-граммы) и char TF-IDF (3–5-граммы), после чего обучается
логистическая регрессия. Модель, метаданные, назначения частей и отложенный test сохраняются
в `artifacts/`.

## Структура каталогов

```text
src/product_category_classification/  # данные, split, модель и CLI
data/                                 # схема и синтетический smoke CSV
tests/                                # модульные и интеграционные тесты
.github/workflows/ci.yml              # CI
```

## Используемые технологии

Python 3.11, pandas, NumPy, scikit-learn, joblib, pytest и Ruff.

## Требования к окружению

Python 3.11 или 3.12; загрузка внешних данных для тестов не нужна.

## Установка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Подготовка данных

Формат и официальный источник описаны в [`data/README.md`](data/README.md). Smoke-набор
воспроизводится командой `product-generate-smoke --output data/smoke.csv`.
Для learning curve генератор принимает `--sellers-per-category` и `--seed`, сохраняя
минимум трёх независимых продавцов на категорию для seller-group split.

## Запуск обучения

```bash
product-train --data data/smoke.csv --output-dir artifacts/smoke
```

## Запуск оценки

```bash
product-evaluate --model artifacts/smoke/model.joblib \
  --data artifacts/smoke/test.csv --output artifacts/smoke/test_metrics.json
```

## Запуск инференса

```bash
product-predict --model artifacts/smoke/model.joblib --text "wireless gaming mouse" --top-k 2
```

## Метрики

Основные метрики — macro-F1 и top-2 accuracy. Macro-F1 одинаково учитывает редкие и
частые категории; top-2 отражает полезность модели для подсказки двух вариантов.

## Тестирование

`pytest -q` проверяет схему, отсутствие пересечения продавцов и полный smoke-сценарий;
`ruff check .` выполняет статическую проверку.

## Инженерные эксперименты

`make experiment` создаёт отдельную synthetic-выборку, выполняет seller-disjoint split
и сохраняет validation/test macro-F1, top-2 accuracy, confusion matrix и per-class метрики.
Параметры `--feature-mode`, `--regularization-c` и границы n-грамм поддерживают абляции,
а `--baseline` добавляет дельты метрик и числа признаков. Smoke-результаты проверяют
pipeline и не являются оценкой реального каталога.
Для data-quality сценариев доступны выбор текстового поля, шум train-меток, токенный шум
на validation/test и явные доли seller-group частей.

## Ограничения

Синтетические примеры не отражают шум реального каталога. Доминирующая категория
используется только для балансировки split, а продавец целиком остаётся в одной части.
Перед эксплуатацией нужны аудит таксономии, мониторинг дрейфа и ручной анализ ошибок.

## Полученные результаты

Численные результаты реального эксперимента не заявляются. CLI фактически рассчитывает
и сохраняет метрики выбранного набора; `smoke.csv` проверяет только работоспособность.

## Источник и лицензия данных

Рекомендуемый открытый источник — [UCI Product Classification and
Clustering](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering),
лицензия CC BY 4.0. В репозитории находится только собственный синтетический набор.

## Статус проекта

Завершён воспроизводимый baseline: подготовка, обучение, оценка и инференс доступны как CLI.
