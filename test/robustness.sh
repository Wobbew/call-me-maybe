#!/usr/bin/env bash
# Robustness checks: the program must never show a traceback and must give
# a clear error message when it exits with an error.
# Run from the repository root:  bash test/robustness.sh
set -u

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FN=data/input/functions_definition.json
IN=data/input/function_calling_tests.json
PASS=0
FAIL=0

# run_case NAME ALLOWED_EXIT_CODES ARGS...
run_case() {
    local name="$1" allowed="$2"
    shift 2
    uv run python -m src --output "$TMP/out.json" "$@" \
        >"$TMP/stdout" 2>"$TMP/stderr"
    local code=$?
    local problems=""
    if grep -q "Traceback" "$TMP/stdout" "$TMP/stderr"; then
        problems="traceback shown;"
    fi
    case " $allowed " in
        *" $code "*) ;;
        *) problems="$problems exit code $code (allowed: $allowed);" ;;
    esac
    if [ "$code" -ne 0 ] && ! grep -q "Error" "$TMP/stderr"; then
        problems="$problems no clear 'Error:' message;"
    fi
    if [ -z "$problems" ]; then
        echo "PASS  $name (exit $code)"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name -> $problems"
        tail -n 4 "$TMP/stderr" | sed 's/^/        /'
        FAIL=$((FAIL + 1))
    fi
}

echo "--- bad input files (fast, model never loads) ---"
run_case "missing functions file" "1" \
    --functions_definition "$TMP/nope.json" --input "$IN"
run_case "missing input file" "1" \
    --functions_definition "$FN" --input "$TMP/nope.json"

echo '{not json' > "$TMP/badjson.json"
run_case "invalid JSON in functions file" "1" \
    --functions_definition "$TMP/badjson.json" --input "$IN"
run_case "invalid JSON in input file" "1" \
    --functions_definition "$FN" --input "$TMP/badjson.json"

echo '{"a": 1}' > "$TMP/notlist.json"
run_case "JSON object instead of list" "1" \
    --functions_definition "$TMP/notlist.json" --input "$IN"

echo '[]' > "$TMP/empty.json"
run_case "empty function list" "1" \
    --functions_definition "$TMP/empty.json" --input "$IN"

echo '[{"nope": "x"}]' > "$TMP/nokey.json"
run_case "prompt entry without 'prompt' key" "1" \
    --functions_definition "$FN" --input "$TMP/nokey.json"

printf '\xff\xfe\x00bad' > "$TMP/binary.json"
run_case "input file with invalid UTF-8" "1" \
    --functions_definition "$FN" --input "$TMP/binary.json"

run_case "directory given instead of a file" "1" \
    --functions_definition "$FN" --input "$TMP"

cat > "$TMP/notype.json" << 'EOF'
[{"name": "fn_x", "description": "d",
  "parameters": {"a": {}}, "returns": {"type": "string"}}]
EOF
echo '[{"prompt": "Do x with 1"}]' > "$TMP/one.json"
run_case "parameter without a type" "0 1" \
    --functions_definition "$TMP/notype.json" --input "$TMP/one.json"

cat > "$TMP/array.json" << 'EOF'
[{"name": "fn_x", "description": "d",
  "parameters": {"a": {"type": "array"}}, "returns": {"type": "string"}}]
EOF
run_case "unsupported parameter type (array)" "0 1" \
    --functions_definition "$TMP/array.json" --input "$TMP/one.json"

echo "--- output problems ---"
run_case "output path cannot be written" "1" \
    --functions_definition "$FN" --input "$TMP/one.json" \
    --output /proc/nope/out.json

echo "--- odd prompts (loads the model, slower) ---"
echo '[]' > "$TMP/noprompts.json"
run_case "empty prompt list" "0" \
    --functions_definition "$FN" --input "$TMP/noprompts.json"

echo '[{"prompt": ""}]' > "$TMP/emptyprompt.json"
run_case "empty prompt string" "0 1" \
    --functions_definition "$FN" --input "$TMP/emptyprompt.json"

echo '[{"prompt": "Greet 😀 ñ 中文"}]' > "$TMP/emoji.json"
run_case "emoji and non-latin text in prompt" "0 1" \
    --functions_definition "$FN" --input "$TMP/emoji.json"

python3 -c "print('[{\"prompt\": \"Greet ' + 'a' * 1500 + '\"}]')" \
    > "$TMP/long.json"
run_case "very long prompt (1500 chars)" "0 1" \
    --functions_definition "$FN" --input "$TMP/long.json"

echo "--- determinism (same input twice -> same output) ---"
head -n 3 "$IN" > /dev/null
python3 - << EOF
import json
data = json.load(open("$IN"))[:3]
json.dump(data, open("$TMP/small.json", "w"))
EOF
run_case "run 1" "0" --functions_definition "$FN" --input "$TMP/small.json"
cp "$TMP/out.json" "$TMP/run1.json"
run_case "run 2" "0" --functions_definition "$FN" --input "$TMP/small.json"
if cmp -s "$TMP/run1.json" "$TMP/out.json"; then
    echo "PASS  outputs are identical"
    PASS=$((PASS + 1))
else
    echo "FAIL  outputs differ between runs"
    FAIL=$((FAIL + 1))
fi

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
