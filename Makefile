.PHONY: install lint test smoke train evaluate predict

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

smoke:
	product-generate-smoke --output data/smoke.csv

train:
	product-train --data data/smoke.csv --output-dir artifacts/smoke

evaluate:
	product-evaluate --model artifacts/smoke/model.joblib --data artifacts/smoke/test.csv --output artifacts/smoke/test_metrics.json

predict:
	product-predict --model artifacts/smoke/model.joblib --text "wireless gaming mouse"
