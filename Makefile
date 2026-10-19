VENV := .venv
PYTHON := $(VENV)/bin/python
PIP_COMPILE := $(VENV)/bin/pip-compile

.PHONY: lint fix-lint compile test run install

lint:
	$(VENV)/bin/ruff check src tests

fix-lint:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

compile:
	$(PIP_COMPILE) --strip-extras -o requirements.txt requirements.in
	$(PIP_COMPILE) --strip-extras -o requirements-dev.txt requirements-dev.in

test:
	$(VENV)/bin/pytest

install:
	$(VENV)/bin/pip install -r requirements-dev.txt -e .

run:
	@set -a; [ -f .env ] && . ./.env; set +a; $(VENV)/bin/agentmem
