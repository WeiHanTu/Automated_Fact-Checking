.PHONY: help install clean test train evaluate data-prepare

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  clean          - Clean generated files"
	@echo "  data-prepare   - Prepare data for training"
	@echo "  train          - Train the model"
	@echo "  evaluate       - Evaluate the model"
	@echo "  test           - Run tests"
	@echo "  format         - Format code with black"
	@echo "  lint           - Run linting with flake8"

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Prepare data
data-prepare:
	python src/main.py --compose

# Train model
train:
	python src/main.py --train --experiment_name "experiment_1"

# Evaluate model
evaluate:
	python src/main.py --evaluate --experiment_name "experiment_1"

# Test model
test-model:
	python src/main.py --test --experiment_name "experiment_1"

# Run tests
test:
	pytest tests/ -v

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Lint code
lint:
	flake8 src/ tests/
	mypy src/

# Create virtual environment
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Install development dependencies
install-dev: install
	pip install -e ".[dev]"

# Run full pipeline
pipeline: data-prepare train evaluate test-model

# Show project structure
tree:
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|venv|.git|wandb|logs' -a

# Create requirements from current environment
freeze:
	pip freeze > requirements.txt

# Update requirements
update-req:
	pip install --upgrade -r requirements.txt 