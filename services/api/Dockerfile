FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install shared library first (must be available before the main package)
COPY libs/text_features /tmp/text_features
RUN pip install --no-cache-dir /tmp/text_features && rm -rf /tmp/text_features

COPY services/api/pyproject.toml ./
COPY services/api/README.md ./
COPY services/api/app ./app

# Install API + ML extras (torch, transformers, lightgbm, etc.)
# This produces a large image (~3-4 GB) but is required for artifact-backed inference.
RUN pip install --no-cache-dir '.[ml]'

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
