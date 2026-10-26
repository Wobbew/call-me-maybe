"""Command-line entry point for the function calling tool."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .llm_interface import LLMInterface
from .models import FunctionDefinition, TestPrompt


class InputError(Exception):
    """Raised when an input file is missing or invalid."""


def load_json_list(path: Path) -> list[Any]:
    """Read a JSON file that must contain a list."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise InputError(f"file not found: {path}")
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not valid JSON: {e}")
    except OSError as e:
        raise InputError(f"cannot read {path}: {e}")
    if not isinstance(data, list):
        raise InputError(f"{path} must contain a JSON list")
    return data


def parse_args() -> argparse.Namespace:
    """Define and read the command-line options."""
    p = argparse.ArgumentParser(description="Function calling tool")
    p.add_argument("--functions_definition", type=Path,
                   default=Path("data/input/functions_definition.json"))
    p.add_argument("--input", type=Path,
                   default=Path("data/input/function_calling_tests.json"))
    p.add_argument("--output", type=Path,
                   default=Path("data/output/function_calling_results.json"))
    return p.parse_args()


def build_context(functions: list[FunctionDefinition], sentence: str) -> str:
    """Build the text the model reads before choosing a function."""
    lines = []
    for f in functions:
        args = ", ".join(f"{n}: {spec['type']}"
                         for n, spec in f.parameters.items())
        lines.append(f"- {f.name}({args}): {f.description}")
    example = ('{"name": "fn_example", '
               '"parameters": {"text": "good morning"}}')
    return (
        "Choose the function that best matches the request and fill in "
        "its arguments. Copy values exactly from the request, without "
        "surrounding quotation marks and without extra spaces.\n\n"
        "Functions:\n" + "\n".join(lines) + "\n\n"
        "Example request: Shout 'good morning'\n"
        f"Example answer: {example}\n\n"
        f"Request: {sentence}\n"
        'Answer: {"name": "'
    )


def process_prompt(llm: LLMInterface,
                   functions: list[FunctionDefinition],
                   sentence: str) -> dict[str, Any]:
    """Pick a function and generate its parameters for one prompt."""
    context = build_context(functions, sentence)
    names = [f.name + '"' for f in functions]   # closing quote ends a name
    picked_raw = llm.choose(context, names)
    picked = picked_raw.rstrip('"')
    function = next(f for f in functions if f.name == picked)
    params = llm.generate_parameters(
        context + picked_raw + ', "parameters": {', function)
    return {"prompt": sentence, "name": picked, "parameters": params}


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Write the results list as JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> int:
    """Load inputs, run the model on each prompt, write the results."""
    args = parse_args()
    try:
        functions = [FunctionDefinition.model_validate(f)
                     for f in load_json_list(args.functions_definition)]
        prompts = [TestPrompt.model_validate(t)
                   for t in load_json_list(args.input)]
    except (InputError, ValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not functions:
        print("Error: no functions defined", file=sys.stderr)
        return 1

    try:
        llm = LLMInterface()
    except Exception as e:  # model loading can fail in many ways
        print(f"Error: could not load the model: {e}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for p in prompts:
        try:
            result = process_prompt(llm, functions, p.prompt)
        except (RuntimeError, ValueError) as e:
            print(f"Error on prompt {p.prompt!r}: {e}", file=sys.stderr)
            continue
        results.append(result)
        print(f"{p.prompt!r} -> {result['name']} {result['parameters']}")

    try:
        write_results(args.output, results)
    except OSError as e:
        print(f"Error: cannot write {args.output}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
    