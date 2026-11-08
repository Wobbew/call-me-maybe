import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


def _unicode_to_byte() -> dict[str, int]:
    """Inverse of the GPT-2 byte -> printable-character table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class LLMInterface(BaseModel):
    """Constrained-decoding wrapper around the small LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    model: Small_LLM_Model | None = None
    id_to_text: dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Load the model and vocabulary after validation."""
        self.model = Small_LLM_Model(self.model_name)
        self.id_to_text = self._load_vocab()

    def _load_vocab(self) -> dict[int, str]:
        """Map token id -> real text by undoing the byte disguise."""
        assert self.model is not None
        table = _unicode_to_byte()
        path = self.model.get_path_to_vocab_file()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        result: dict[int, str] = {}
        for text, token_id in raw.items():
            if not all(c in table for c in text):
                continue
            data = bytes(table[c] for c in text)
            try:
                result[token_id] = data.decode("utf-8")
            except UnicodeDecodeError:
                continue    # half of a multi-byte character: not usable
        return result

    def get_logits(self, text: str) -> list[float]:
        """Return next-token scores for the text."""
        assert self.model is not None
        ids = self.model.encode(text)[0].tolist()
        return self.model.get_logits_from_input_ids(ids)

    def choose(self, context: str, options: list[str]) -> str:
        """Generate token by token.

        Only tokens that keep the text on the way to one of the options
        are allowed.
        """
        generated = ""
        while True:
            matching = [o for o in options if o.startswith(generated)]
            if not matching:
                raise RuntimeError("no option matches the generated text")
            if len(matching) == 1:
                return matching[0]
            logits = self.get_logits(context + generated)
            best_id, best_score = -1, float("-inf")
            ranked: list[tuple[float, str]] = []        
            for token_id, text in self.id_to_text.items():
                if token_id >= len(logits) or not text:
                    continue
                if any(o.startswith(generated + text) for o in matching):
                    ranked.append((logits[token_id], text))
                    if logits[token_id] > best_score:
                        best_id, best_score = token_id, logits[token_id]
            if " true" in options:
                ranked.sort(reverse=True)
                print("DEBUG", repr(generated), ranked[:3])
            if best_id == -1:
                raise RuntimeError("no valid token available")
            generated += self.id_to_text[best_id]

    @staticmethod
    def _ends_escaped(s: str) -> bool:
        """Return True if s ends with an odd number of backslashes."""
        return (len(s) - len(s.rstrip("\\"))) % 2 == 1

    def generate_string(self, context: str, max_tokens: int = 100) -> str:
        """Generate string content until the model closes the quote."""
        value = ""
        for _ in range(max_tokens):
            logits = self.get_logits(context + value)
            best_id, best_score = -1, float("-inf")
            for token_id, text in self.id_to_text.items():
                if token_id >= len(logits) or not text or "\n" in text:
                    continue
                if logits[token_id] > best_score:
                    best_id, best_score = token_id, logits[token_id]
            if best_id == -1:
                break
            text = self.id_to_text[best_id]
            if '"' in text:
                head = text.split('"', 1)[0]
                if self._ends_escaped(value + head):
                    value += text          # escaped quote: keep going
                    continue
                value += head
                break
            value += text
        return value.strip().replace('\\"', '"')

    def generate_number(self, context: str, max_tokens: int = 20) -> str:
        """Generate a number; the context must end right after the colon."""
        value = ""
        for _ in range(max_tokens):
            logits = self.get_logits(context + value)
            body = value.strip()
            ends_digit = body[-1:] in list("0123456789")
            best_id, best_score = -1, float("-inf")
            for token_id, tok in self.id_to_text.items():
                if token_id >= len(logits):
                    continue
                allowed = (
                    (len(tok) == 1 and tok in "0123456789")
                    or (tok in (" ", " -") and value == "")
                    or (tok == "-" and body == "")
                    or (tok == "." and ends_digit and "." not in body)
                    or (tok in (",", "}") and ends_digit)
                )
                if allowed and logits[token_id] > best_score:
                    best_id, best_score = token_id, logits[token_id]
            if best_id == -1 or self.id_to_text[best_id] in (",", "}"):
                break
            value += self.id_to_text[best_id]
        return value

    def generate_parameters(self, context: str,
                            function: FunctionDefinition) -> dict[str, Any]:
        """Generate every parameter of the chosen function."""
        params: dict[str, Any] = {}
        text = context
        names = list(function.parameters)

        for i, name in enumerate(names):
            kind = function.parameters[name]["type"]
            text += f'"{name}":'
            if kind == "string":
                text += ' "'
                s = self.generate_string(text)
                params[name] = s
                text += s + '"'
            elif kind in ("number", "integer"):
                raw = self.generate_number(text)
                try:
                    num = float(raw)
                except ValueError:
                    raise RuntimeError(f"bad number for '{name}': {raw!r}")
                params[name] = int(num) if kind == "integer" else num
                text += " " + raw.strip()
            elif kind == "boolean":
                print(text)
                raw = self.choose(text, [" true", " false"])
                print(raw)
                params[name] = raw.strip() == "true"
                text += raw
            else:
                raise RuntimeError(f"unsupported type '{kind}' for '{name}'")
            text += ", " if i < len(names) - 1 else "}"
        return params
