"""Check a results file against the function definitions.

Usage:
    python check_results.py DEFINITIONS PROMPTS RESULTS [EXPECTED_JSON]

EXPECTED_JSON is optional: {"prompt": {"name": ..., "parameters": {...}}
or null}. Without it, the built-in EXPECTED table is used.

Example:
    python check_results.py data/input/functions_definition.json \
        data/input/edge_tests.json /tmp/out.json

Two levels of checking:
  * STRUCTURE (must always pass): one entry per prompt, exact keys, known
    function name, exact parameter names, correct JSON types.
  * ACCURACY (informational): compares with the values listed in EXPECTED.
    Only the keys listed are compared. None means "no right answer, just
    don't crash".
"""
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED: dict[str, tuple[str, dict[str, Any]] | None] = {
    "What is the sum of 2 and 3?":
        ("fn_add_numbers", {"a": 2, "b": 3}),
    "What is the sum of 1000000000 and 2500000000?":
        ("fn_add_numbers", {"a": 1000000000, "b": 2500000000}),
    "What is the sum of -5 and 3.5?":
        ("fn_add_numbers", {"a": -5, "b": 3.5}),
    "Greet shrek":
        ("fn_greet", {"name": "shrek"}),
    "Greet josé":
        ("fn_greet", {"name": "josé"}),
    "Greet John Smith":
        ("fn_greet", {"name": "John Smith"}),
    "Reverse the string ''":
        ("fn_reverse_string", {"s": ""}),
    "Reverse the string 'hello'":
        ("fn_reverse_string", {"s": "hello"}),
    "Reverse the string 'héllo wörld'":
        ("fn_reverse_string", {"s": "héllo wörld"}),
    # Known limit: generate_string stops at the first double quote.
    "Reverse the string 'say \"hi\"'":
        ("fn_reverse_string", {"s": "say \"hi\""}),
    "What is the square root of 16?":
        ("fn_get_square_root", {"a": 16}),
    "Calculate the square root of 2.25":
        ("fn_get_square_root", {"a": 2.25}),
    "Replace all digits in 'a1b2c3' with X":
        ("fn_substitute_string_with_regex",
         {"source_string": "a1b2c3", "replacement": "X"}),
    "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'":
        ("fn_substitute_string_with_regex",
         {"source_string": "The cat sat on the mat",
          "regex": "cat", "replacement": "dog"}),
    "Replace all vowels in 'Programming is fun' with asterisks":
        ("fn_substitute_string_with_regex",
         {"source_string": "Programming is fun"}),
    "Greet bob and reverse it": None,
    "What is the sum of apples and 3?": None,
    "Tell me a joke": None,
}

PY_TYPES: dict[str, tuple[type, ...]] = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "boolean": (bool,),
}


def load(path: str) -> Any:
    """Load a JSON file or exit with a clear message."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"cannot load {path}: {e}")
        sys.exit(2)


def type_ok(value: Any, kind: str) -> bool:
    """Check a JSON value against a schema type (bool is not a number)."""
    if isinstance(value, bool) and kind != "boolean":
        return False
    return isinstance(value, PY_TYPES.get(kind, ()))


def check_structure(entry: dict[str, Any],
                    defs: dict[str, dict[str, Any]]) -> list[str]:
    """Return a list of structural problems for one result entry."""
    problems: list[str] = []
    if set(entry) != {"prompt", "name", "parameters"}:
        problems.append(f"keys are {sorted(entry)}")
        return problems
    func = defs.get(entry["name"])
    if func is None:
        problems.append(f"unknown function {entry['name']!r}")
        return problems
    schema = func["parameters"]
    params = entry["parameters"]
    if set(params) != set(schema):
        problems.append(
            f"parameter names {sorted(params)} != {sorted(schema)}")
        return problems
    for name, spec in schema.items():
        if not type_ok(params[name], spec["type"]):
            problems.append(
                f"{name}={params[name]!r} is not {spec['type']}")
    return problems


def check_accuracy(entry: dict[str, Any]) -> str:
    """Compare one entry with EXPECTED; return a status string."""
    if entry["prompt"] not in EXPECTED:
        return "no expectation"
    expected = EXPECTED[entry["prompt"]]
    if expected is None:
        return "no right answer (only structure matters)"
    name, wanted = expected
    diffs: list[str] = []
    if entry["name"] != name:
        diffs.append(f"name {entry['name']} != {name}")
    for key, value in wanted.items():
        got = entry["parameters"].get(key)
        if got != value:
            diffs.append(f"{key}: got {got!r}, wanted {value!r}")
    return "OK" if not diffs else "DIFF: " + "; ".join(diffs)


def main() -> int:
    """Run all checks and print a report."""
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        return 2
    if len(sys.argv) == 5:
        EXPECTED.clear()
        for prompt, exp in load(sys.argv[4]).items():
            EXPECTED[prompt] = (
                None if exp is None else (exp["name"], exp["parameters"])
            )
    defs_list = load(sys.argv[1])
    prompts = load(sys.argv[2])
    results = load(sys.argv[3])
    defs = {d["name"]: d for d in defs_list}

    by_prompt = {r.get("prompt"): r for r in results if isinstance(r, dict)}
    structure_fail = 0
    accurate = 0
    scored = 0

    for item in prompts:
        prompt = item["prompt"]
        entry = by_prompt.get(prompt)
        print(f"\n> {prompt}")
        if entry is None:
            print("  STRUCTURE: MISSING from output")
            structure_fail += 1
            continue
        problems = check_structure(entry, defs)
        if problems:
            structure_fail += 1
            print("  STRUCTURE: FAIL " + "; ".join(problems))
        else:
            print("  STRUCTURE: ok")
        print(f"  got: {entry.get('name')} {entry.get('parameters')}")
        if not problems:
            status = check_accuracy(entry)
            print(f"  ACCURACY: {status}")
            if status == "OK":
                accurate += 1
                scored += 1
            elif status.startswith("DIFF"):
                scored += 1

    print("\n" + "=" * 50)
    print(f"prompts: {len(prompts)}  results: {len(results)}")
    print(f"structure failures: {structure_fail}")
    if scored:
        print(f"accuracy on scored prompts: {accurate}/{scored} "
              f"({100 * accurate // scored}%)")
    return 1 if structure_fail else 0


if __name__ == "__main__":
    sys.exit(main())
