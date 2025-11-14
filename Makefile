# Ingeniero: Stephanie Poleo
# Proyecto: Curso ETL - Computer Vision
# Nov. 2025

# Este Makefile permite automatizar la creación del entorno virtual,
# instalación de librerías, linting con pylint, formateo del código
# y ejecución de pruebas unitarias.

# =========================
# PYTHON / ENTORNO VIRTUAL
# =========================
PYTHON_BIN       := py -3.10
ENV_PATH         := venv
VENV_PY          := $(ENV_PATH)\Scripts\python.exe
VENV_PIP         := $(ENV_PATH)\Scripts\pip3.10.exe
REQUIREMENTS_FILE := requirements.txt

# =========================
# TARGETS
# =========================

.PHONY: help
help:
	@echo "Comandos disponibles:"
	@echo "  make venv        - Crear entorno virtual Python (si no existe)"
	@echo "  make install     - Instalar dependencias (incluye venv)"
	@echo "  make format      - Formatear código con black"
	@echo "  make lint        - Linting con pylint"
	@echo "  make test        - Ejecutar pruebas unitarias"
	@echo "  make all         - Pipeline completo (install, format, lint, test)"

.PHONY: venv
venv:
	@if exist "$(VENV_PY)" ( \
		echo ^>^> Entorno virtual ya existe en $(ENV_PATH). No se recrea. \
	) else ( \
		echo ^>^> Creando entorno virtual en $(ENV_PATH) && \
		$(PYTHON_BIN) -m venv "$(ENV_PATH)" && \
		if exist "$(VENV_PY)" ( \
			echo ^>^> Entorno virtual creado OK. \
		) else ( \
			echo ERROR: venv no se creo correctamente en $(ENV_PATH) && exit /b 1 \
		) \
	)

.PHONY: install
install: venv
	@echo ^>^> Instalando dependencias desde $(REQUIREMENTS_FILE)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install -r "$(REQUIREMENTS_FILE)"

.PHONY: format
format:
	@echo ^>^> Formateando codigo con black
	@$(VENV_PY) -m black .

.PHONY: lint
lint:
	@echo ^>^> Linting con pylint
	@$(VENV_PY) -m pylint --disable=R,C *.py

.PHONY: test
test:
	@echo ^>^> Ejecutando pruebas unitarias
	@$(VENV_PY) -m pytest -vv


.PHONY: all
all: install format lint test
	@echo ^>^> Pipeline completo OK