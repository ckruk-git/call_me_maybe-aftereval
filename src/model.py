from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    type: str


class ReturnDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


model = Small_LLM_Model()


def get_masked_logits(
    logits: list[float],
    allowed_tokens: list[int],
) -> list[float]:
    if not allowed_tokens:
        return logits
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    return list(mask)
