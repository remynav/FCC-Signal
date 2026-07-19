.PHONY: install test lint pull features evaluate

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

pull:            ## scheduled ingestion (also run by Prefect)
	python -m fccsignal.ingestion.run_pulls

features:        ## rebuild the PIT feature store from the raw zone
	python -m fccsignal.features.build

evaluate:        ## run the pre-registered spec grid
	python -m fccsignal.evaluation.run_grid --config configs/spec_grid.yaml
