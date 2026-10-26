from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    """One function the model may choose."""

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class TestPrompt(BaseModel):
    """One natural-language request."""

    prompt: str
