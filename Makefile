PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: install-web install-api install-ml dev-web dev-api test-api build-web build-trainer

install-web:
	cd apps/web && npm install

install-api:
	python3 -m venv .venv
	cd services/api && ../../.venv/bin/python -m pip install -e '.[dev]'

dev-web:
	cd apps/web && npm run dev

dev-api:
	cd services/api && ../../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test-api:
	cd services/api && ../../.venv/bin/pytest

build-web:
	cd apps/web && npm run build

install-ml:
	python3 -m venv .venv-ml
	cd services/ml && ../../.venv-ml/bin/python -m pip install -e .

build-trainer:
	docker build -t ai-text-detector-trainer:latest services/ml
