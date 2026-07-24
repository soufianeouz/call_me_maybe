install:
	uv sync

run:
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/tests.json --output data/output/results.json

debug:
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/tests.json --output data/output/results.json

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__