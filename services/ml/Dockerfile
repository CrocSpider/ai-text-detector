FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Install shared library first (must be available before the main package)
COPY libs/text_features /tmp/text_features
RUN pip install /tmp/text_features && rm -rf /tmp/text_features

COPY services/ml/pyproject.toml services/ml/README.md ./
COPY services/ml/trainer ./trainer
COPY services/ml/configs ./configs

RUN pip install --upgrade pip && pip install .

ENTRYPOINT ["python", "-m", "trainer.cli"]
CMD ["--help"]
