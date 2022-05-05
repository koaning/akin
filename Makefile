black:
	black akin tests setup.py --check

flake:
	flake8 akin tests setup.py

test:
	pytest tests

check: black flake test

install:
	python -m pip install -e ".[dev]"

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache
