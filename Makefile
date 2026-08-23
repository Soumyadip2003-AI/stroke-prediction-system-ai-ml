PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
FRONTEND_DIR := neuropredict-frontend

.PHONY: setup setup-python setup-frontend install dev backend frontend check test

setup: setup-python setup-frontend

setup-python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-frontend:
	npm --prefix $(FRONTEND_DIR) install

dev:
	$(PYTHON_VENV) backend.py & npm --prefix $(FRONTEND_DIR) start

backend:
	$(PYTHON_VENV) backend.py

frontend:
	npm --prefix $(FRONTEND_DIR) start

check:
	$(PYTHON_VENV) -m compileall .
	npm --prefix $(FRONTEND_DIR) run build

test:
	npm --prefix $(FRONTEND_DIR) test -- --watch=false
