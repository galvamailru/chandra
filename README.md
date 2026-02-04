# Chandra OCR Service

HTTP-сервис OCR на базе [Chandra](https://github.com/datalab-to/chandra) (datalab-to/chandra): распознавание документов с таблицами, формулами, рукописным текстом и сложной вёрсткой.

## Запуск в Docker

```bash
docker compose up -d --build
```

Сервис будет доступен на **http://localhost:8002**.

- **GET /** — веб-интерфейс загрузки файла и просмотра результата (Markdown / HTML).
- **GET /healthz** — проверка состояния сервиса.
- **POST /parse** — загрузка PDF или изображения; в ответе — `markdown`, `html`, метаданные по страницам.

### Пример запроса к API

```bash
curl -X POST http://localhost:8002/parse -F "file=@document.pdf"
```

Опционально: диапазон страниц для PDF — query-параметр `page_range`, например `?page_range=1-5,7,9-12`.

## Локальный запуск (без Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

При первом запросе будет загружена модель (HuggingFace). Для ускорения можно предзагрузить:

```bash
python -c "from chandra.model import InferenceManager; InferenceManager(method='hf')"
```

## Режимы inference

- **hf** (по умолчанию в этом сервисе) — локальный запуск через HuggingFace Transformers в одном контейнере.
- **vLLM** — для продакшена и высокой нагрузки: отдельно поднимается vLLM-сервер (`chandra_vllm`), затем можно изменить код сервиса на `InferenceManager(method="vllm")` и указать `VLLM_API_BASE`.

## Переменные окружения

- `MODEL_CHECKPOINT` — чекпоинт модели (по умолчанию `datalab-to/chandra`).
- `MAX_OUTPUT_TOKENS` — максимум токенов на страницу (по умолчанию из настроек chandra).

## Лицензия

Код Chandra — Apache 2.0. Веса модели — [MODEL_LICENSE](https://github.com/datalab-to/chandra/blob/master/MODEL_LICENSE) (ограничения на коммерческое использование).
