import json
from typing import Any
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


class LLMInterface:

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the model and build the id -> text vocabulary."""
        self.model = Small_LLM_Model(model_name)
        self.id_to_text = self._load_vocab()

    def _load_vocab(self) -> dict[int, str]:
        """Invert vocab.json and fix the byte markers."""
        with open(self.model.get_path_to_vocab_file()) as f:
            raw = json.load(f)
        return {
            token_id: text.replace("Ġ", " ").replace("Ċ", "\n")
            for text, token_id in raw.items()
        }

    def get_logits(self, text: str) -> list[float]:
        """Return next-token scores for the text."""
        ids = self.model.encode(text)[0].tolist()
        return self.model.get_logits_from_input_ids(ids)

    def choose(self, context: str, options: list[str]) -> str:
        """Generate token by token, only allowing text that leads to an option."""
        generated = ""
        while True:

            matching = [o for o in options if o.startswith(generated)]
            if not matching:
                raise RuntimeError("no option matches the generated text")
            if len(matching) == 1:          # autocomplete, skip the model
                return matching[0]
            logits = self.get_logits(context + generated)
            best_id, best_score = -1, float("-inf")
            for token_id, text in self.id_to_text.items():
                if token_id >= len(logits) or not text:
                    continue
                if any(o.startswith(generated + text) for o in matching):
                    if logits[token_id] > best_score:
                        best_id, best_score = token_id, logits[token_id]
            if best_id == -1:
                raise RuntimeError("no valid token available")
            generated += self.id_to_text[best_id]

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
                value += text.split('"', 1)[0]
                break
            value += text
        return value

    def generate_number(self, context: str, max_tokens: int = 20) -> str:
        """Generate a number's characters until ',' or '}' is chosen."""
        value = ""
        for _ in range(max_tokens):
            logits = self.get_logits(context + value)
            ends_digit = value[-1:] in list("0123456789")
            best_id, best_score = -1, float("-inf")
            for token_id, tok in self.id_to_text.items():
                if token_id >= len(logits):
                    continue
                allowed = (
                    (len(tok) == 1 and tok in "0123456789")
                    or (tok == "-" and value == "")
                    or (tok == "." and ends_digit and "." not in value)
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
            text += f'"{name}": '
            if kind == "string":
                text += '"'
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
                text += raw
            elif kind == "boolean":
                raw = self.choose(text, ["true", "false"])
                params[name] = raw == "true"
                text += raw
            else:
                raise RuntimeError(f"unsupported type '{kind}' for '{name}'")
            text += ", " if i < len(names) - 1 else "}"
        return params