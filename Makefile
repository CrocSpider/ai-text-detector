PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: install-web install-api install-ml dev-web dev-api test-api test-ml build-web build-api build-trainer

install-web:
	cd apps/web && npm install

install-api:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e libs/text_features
	cd services/api && ../../.venv/bin/python -m pip install -e '.[dev]'

dev-web:
	cd apps/web && npm run dev

dev-api:
	cd services/api && ../../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test-api:
	cd services/api && ../../.venv/bin/pytest

test-ml:
	cd services/ml && ../../.venv-ml/bin/pytest

build-web:
	cd apps/web && npm run build

build-api:
	docker build -f services/api/Dockerfile -t ai-text-origin-risk-api:latest .

install-ml:
	python3 -m venv .venv-ml
	.venv-ml/bin/python -m pip install -e libs/text_features
	cd services/ml && ../../.venv-ml/bin/python -m pip install -e '.[dev]'

build-trainer:
	docker build -f services/ml/Dockerfile -t ai-text-detector-trainer:latest .
