# Makefile for CCTV Human Tracking project

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies using Poetry"
	@echo "  make clean     - Remove the virtual environment"

# Install dependencies
# Checks if pyproject.toml or poetry.lock is newer than a potential marker file (not strictly necessary for poetry but good practice)
.PHONY: install
install: pyproject.toml poetry.lock
	@echo "Installing dependencies using Poetry..."
	poetry install

# Clean the virtual environment
.PHONY: clean
clean:
	@echo "Removing virtual environment managed by Poetry..."
	poetry env remove python || true # Use '|| true' to avoid error if env doesn't exist
	@echo "Cleaning complete."

# Declare files as prerequisites to potentially trigger install if they change
pyproject.toml:
poetry.lock: