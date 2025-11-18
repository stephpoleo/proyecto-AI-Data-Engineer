# Ingeniero: Stephanie Poleo
# Proyecto: Curso ETL - Computer Vision
# Versión WSL / Linux

PYTHON_BIN        := python3.10
ENV_PATH          := venv
VENV_PY           := $(ENV_PATH)/bin/python
VENV_PIP          := $(ENV_PATH)/bin/pip
REQUIREMENTS_FILE := requirements.txt

.PHONY: help
help:
	@echo "Comandos disponibles:"
	@echo "  make venv        - Crear entorno virtual Python (si no existe)"
	@echo "  make install     - Instalar dependencias (incluye venv)"
	@echo "  make format      - Formatear código con black"
	@echo "  make lint        - Linting con pylint"
	@echo "  make test        - Ejecutar pruebas unitarias (pytest)"
	@echo "  make all         - Pipeline completo (install, format, lint, test)"

# Crear venv solo si no existe
.PHONY: venv
venv:
	@if [ -d "$(ENV_PATH)" ]; then \
		echo ">> Entorno virtual ya existe en $(ENV_PATH). No se recrea."; \
	else \
		echo ">> Creando entorno virtual en $(ENV_PATH)"; \
		$(PYTHON_BIN) -m venv "$(ENV_PATH)"; \
	fi

.PHONY: install
install: venv
	@echo ">> Instalando dependencias desde $(REQUIREMENTS_FILE)"
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install -r "$(REQUIREMENTS_FILE)"

.PHONY: format
format:
	@echo ">> Formateando código con black"
	@$(VENV_PY) -m black src tests main.py

.PHONY: lint
lint:
	@echo ">> Linting con pylint"
	@$(VENV_PY) -m pylint \
	    src/ \
	    tests/ \
	    main.py \
	    --ignore=venv \
	    -disable=R,C,R0801 || true

.PHONY: test
test:
	@echo ">> Ejecutando pruebas unitarias"
	@$(VENV_PY) -m pytest -vv

.PHONY: all
all: install format lint test
	@echo ">> Pipeline completo OK"
