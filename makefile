ARGS ?=

.PHONY: all install run debug clean fclean lint lint-strict

all: install

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache

fclean: clean
	rm -rf .venv

lint:
	uv run flake8 . --exclude .venv,llm_sdk
	uv run mypy . --exclude '(\.venv|llm_sdk)' --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude .venv,llm_sdk
	uv run mypy . --exclude '(\.venv|llm_sdk)' --strict