PYTHON ?= $(shell which python)
SRC ?= transformer examples

.PHONY: lint
lint:
	$(PYTHON) -m black --check --line-length=120 $(SRC)
	$(PYTHON) -m flake8 $(SRC)
	$(PYTHON) -m isort -rc -c $(SRC)
	$(PYTHON) -m mypy $(SRC)

.PHONY: format
format:
	$(PYTHON) -m isort -rc $(SRC)
	$(PYTHON) -m black --line-length=120 $(SRC)
