"""Unit tests that need no real model: parsing, IO, tokenizer table, decoding.

Run from the repository root:
    uv run --with pytest pytest test/test_units.py

If your entry file is not src/__main__.py, change the import below.
The decoding tests use a fake model whose "favourite" next token is
scripted, so they check that the constraints (not the model) keep the
output valid.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.llm_interface as li
from src.__main__ import (
    InputError,
    build_context,
    load_json_list,
    write_results,
)
from src.models import FunctionDefinition

VOCAB = {
    "f": 0, "n": 1, "_": 2, "x": 3, "y": 4, '"': 5, "1": 6, "2": 7,
    "-": 8, ".": 9, ",": 10, "}": 11, "Ġ": 12, "z": 13, "Ã": 14,
    "Ã©": 15, "h": 16, "i": 17, "e": 18, "l": 19, "o": 20, "w": 21,
    "r": 22, "d": 23, "5": 24,
}


class FakeIds:
    """Mimics the tensor row returned by encode()."""

    def __init__(self, ids: list[int]) -> None:
        """Keep the ids."""
        self.ids = ids

    def tolist(self) -> list[int]:
        """Return the ids as a list."""
        return self.ids


class Script:
    """What the fake model wants to write next."""

    ctx = ""
    target = ""
    tempt = ""


def script(ctx: str, target: str, tempt: str = "") -> None:
    """Tell the fake model what to write after ctx.

    tempt is a token that gets a huge score on the first step only, to
    check that the constraints refuse it.
    """
    Script.ctx, Script.target, Script.tempt = ctx, target, tempt


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> li.LLMInterface:
    """Build an LLMInterface on top of a scripted fake model."""
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(VOCAB), encoding="utf-8")

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_path_to_vocab_file(self) -> str:
            return str(vocab_file)

        def encode(self, text: str) -> list[FakeIds]:
            return [FakeIds([ord(c) for c in text])]

        def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
            text = "".join(chr(i) for i in ids)
            done = text[len(Script.ctx):]
            scores = [0.0] * len(VOCAB)
            want = Script.target[len(done):len(done) + 1]
            key = "Ġ" if want == " " else want
            if key in VOCAB:
                scores[VOCAB[key]] = 1.0
            if Script.tempt and not done:
                scores[VOCAB[Script.tempt]] = 5.0
            return scores

    monkeypatch.setattr(li, "Small_LLM_Model", FakeModel)
    script("", "")
    return li.LLMInterface()


# ---------------------------------------------------------- tokenizer table
def test_byte_table_is_a_bijection() -> None:
    """Every byte has exactly one disguise character."""
    table = li._unicode_to_byte()
    assert len(table) == 256
    assert set(table.values()) == set(range(256))


def test_byte_table_known_values() -> None:
    """The two markers you already knew, plus a UTF-8 round trip."""
    table = li._unicode_to_byte()
    assert table["Ġ"] == 32
    assert table["Ċ"] == 10
    data = bytes(table[c] for c in "Ã©")
    assert data.decode("utf-8") == "é"


def test_vocab_decodes_bytes(llm: li.LLMInterface) -> None:
    """Space, accents and half-characters in the vocabulary."""
    assert llm.id_to_text[VOCAB["Ġ"]] == " "
    assert llm.id_to_text[VOCAB["Ã©"]] == "é"
    assert llm.id_to_text.get(VOCAB["Ã"], "") == ""


# ------------------------------------------------------------------ choose
def test_choose_ignores_invalid_token(llm: li.LLMInterface) -> None:
    """The model prefers 'z', but no option contains it."""
    script("Answer: ", 'fn_y"', tempt="z")
    assert llm.choose("Answer: ", ['fn_x"', 'fn_y"']) == 'fn_y"'


def test_choose_prefix_longer_option(llm: li.LLMInterface) -> None:
    """The closing quote lets fn_xy win over its prefix fn_x."""
    script("A: ", 'fn_xy"')
    assert llm.choose("A: ", ['fn_x"', 'fn_xy"']) == 'fn_xy"'


def test_choose_prefix_shorter_option(llm: li.LLMInterface) -> None:
    """The closing quote also lets the short name end correctly."""
    script("A: ", 'fn_x"')
    assert llm.choose("A: ", ['fn_x"', 'fn_xy"']) == 'fn_x"'


def test_choose_no_options_raises(llm: li.LLMInterface) -> None:
    """Impossible constraints give a clear error, not an endless loop."""
    script("A: ", "")
    with pytest.raises(RuntimeError):
        llm.choose("A: ", [])


# ----------------------------------------------------------------- numbers
def test_number_cannot_start_with_dot(llm: li.LLMInterface) -> None:
    """The model prefers '.', the constraints force a digit first."""
    script("n:", "5,", tempt=".")
    assert llm.generate_number("n:").strip() == "5"


def test_number_negative_decimal(llm: li.LLMInterface) -> None:
    """Sign and decimal point are accepted in the right places."""
    script("n:", "-5.5,")
    assert llm.generate_number("n:").strip() == "-5.5"


# ----------------------------------------------------------------- strings
def test_string_stops_at_quote(llm: li.LLMInterface) -> None:
    """The closing quote ends the value and is not included."""
    script('s: "', 'hello world"')
    assert llm.generate_string('s: "') == "hello world"


def test_string_can_be_empty(llm: li.LLMInterface) -> None:
    """An immediate quote gives the empty string."""
    script('s: "', '"')
    assert llm.generate_string('s: "') == ""


# ------------------------------------------------------------ files and IO
def test_load_json_list_ok(tmp_path: Path) -> None:
    """A valid list is returned unchanged."""
    p = tmp_path / "a.json"
    p.write_text('[{"prompt": "x"}]', encoding="utf-8")
    assert load_json_list(p) == [{"prompt": "x"}]


@pytest.mark.parametrize("content", ["{not json", '{"a": 1}', "", "null"])
def test_load_json_list_rejects_bad_content(
        tmp_path: Path, content: str) -> None:
    """Invalid JSON or a non-list gives InputError, not a traceback."""
    p = tmp_path / "a.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(InputError):
        load_json_list(p)


def test_load_json_list_missing_file(tmp_path: Path) -> None:
    """A missing file gives InputError."""
    with pytest.raises(InputError):
        load_json_list(tmp_path / "nope.json")


def test_load_json_list_invalid_utf8(tmp_path: Path) -> None:
    """Binary garbage gives InputError (UnicodeDecodeError must be caught)."""
    p = tmp_path / "a.json"
    p.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(InputError):
        load_json_list(p)


def test_write_results_creates_folder(tmp_path: Path) -> None:
    """Nested output folders are created and non-ASCII text is kept."""
    out = tmp_path / "deep" / "dir" / "out.json"
    write_results(out, [{"prompt": "josé", "name": "fn", "parameters": {}}])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["prompt"] == "josé"
    assert "josé" in out.read_text(encoding="utf-8")


# ----------------------------------------------------- models and context
def make_function(name: str = "fn_a") -> FunctionDefinition:
    """Build a small valid function definition."""
    return FunctionDefinition.model_validate({
        "name": name,
        "description": "does a",
        "parameters": {"x": {"type": "number"}},
        "returns": {"type": "number"},
    })


def test_function_definition_requires_returns() -> None:
    """A definition without 'returns' is rejected."""
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {"name": "f", "description": "d", "parameters": {}})


def test_build_context_shape() -> None:
    """Context lists every function and ends where the name goes."""
    funcs = [make_function("fn_a"), make_function("fn_b")]
    ctx = build_context(funcs, "do the thing")
    assert "fn_a" in ctx and "fn_b" in ctx
    assert "Request: do the thing" in ctx
    assert ctx.endswith('Answer: {"name": "')
