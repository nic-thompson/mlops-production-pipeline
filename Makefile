.PHONY: help install test coverage drift clean

help:
	@echo "Available commands:"
	@echo "  make install   Install dependencies"
	@echo "  make test      Run unit tests"
	@echo "  make coverage  Run tests with coverage"
	@echo "  make drift     Run drift detection job"
	@echo "  make clean     Remove caches and reports"

install:
	pip install -r requirements.txt

test:
	pytest -v

coverage:
	pytest --cov=src --cov-report=term-missing

drift:
	python -m jobs.drift_job \
		--reference data/reference.parquet \
		--current data/current.parquet \
		--threshold 0.2 \
		--output reports/drift_report.json

clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf reports/*.json

pipeline:
	python -m training.pipeline