.PHONY: build run test lint clean

IMAGE   := legal-doc-analyzer
PORT    := 8501

## build        — Build Docker image
build:
	docker build -t $(IMAGE) .

## run          — Run the app in Docker (foreground)
run: build
	docker compose up

## test         — Run pytest suite locally
test:
	python -m pytest tests/ -v --tb=short

## lint         — Lint with ruff
lint:
	ruff check src/ tests/ utils/

## clean        — Remove build artefacts
clean:
	docker compose down -v 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info dist build
