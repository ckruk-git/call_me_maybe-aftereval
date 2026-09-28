from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

def get_masked_logits(logits, allowed_tokens) -> list[float]:
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    return list(mask)