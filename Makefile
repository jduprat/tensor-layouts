.PHONY: build clean test

build:
	pip install -e .

clean:
	rm -rf build/ dist/ src/*.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf examples_output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete

test:
	python -m pytest tests/ -v
