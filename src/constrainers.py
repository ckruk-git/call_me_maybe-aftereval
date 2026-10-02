from .utils import get_current_parameter, left_parameters
from .model import model, get_masked_logits, FunctionDefinition
import re
from .finder import output_state
from typing import Any


def constrainer(
    output: str,
    logits: list[float],
    function: FunctionDefinition,
    prompt: str,
        ) -> Any:
    state = output_state(output, function)
    print("STATE:", state)
    tokens = []
    if state == "start":
        tokens.append(model.encode('{')[0].tolist()[0])
        return get_masked_logits(logits, tokens)
    if state == "key_start" or state == "key_end" or state == "value_end":
        tokens.append(model.encode('"')[0].tolist()[0])
        return get_masked_logits(logits, tokens)
    if state == "key_content":
        left_par = left_parameters(function, output)
        tokens.extend(model.encode(left_par)[0].tolist())
        return get_masked_logits(logits, tokens)
    if state == "colon":
        tokens.append(model.encode(':')[0].tolist()[0])
        return get_masked_logits(logits, tokens)
    if state == "coma":
        tokens.append(model.encode(',')[0].tolist()[0])
        return get_masked_logits(logits, tokens)
    if state == "over":
        tokens.append(model.encode('}')[0].tolist()[0])
        return get_masked_logits(logits, tokens)
    if state == "value":
        return value_constrainer(function, logits, prompt, output)
    return "error"


def ints_constrained(
        logits: list[float],
        prompt: str,
        output: str,
        function: FunctionDefinition,
        ) -> Any:
    tokens = []
    res = output.split(":", 1)[0]
    prompt_splited = prompt.split()
    allowed = ['}']
    current_parameter = get_current_parameter(output, function)
    if current_parameter != list(function.parameters.keys())[-1]:
        allowed.extend(",")
    for i in prompt_splited:
        numbers = re.findall(r'\d+', i)
        if numbers:
            allowed.extend(numbers)
    print("Allowed", allowed)
    for i in allowed:
        if i not in res:
            tokens.extend(model.encode(i)[0].tolist())
    return get_masked_logits(logits, tokens)


def floats_constrained(logits: list[float], prompt: str) -> Any:
    prompt_splited = prompt.split()
    allowed = ["}", ",", "-", ".", " "]
    for i in range(10):
        allowed.append(str(i))
    tokens = []
    for token in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    for token in prompt_splited:
        tokens.extend(model.encode(i)[0].tolist())
    return get_masked_logits(logits, tokens)


def bool_constrained(logits: list[float]) -> Any:
    allowed = ["true", "false", "}", "'", " "]
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    return get_masked_logits(logits, tokens)


def string_constrained(
        logits: list[float],
        prompt: str,
        output: str,
        function: FunctionDefinition,
        ) -> Any:
    current_parameter = get_current_parameter(output, function)
    tokens = []
    res = output.rsplit(":", 1)[1]
    tokens.extend(model.encode('"')[0].tolist())
    if len(res) == 0:
        return get_masked_logits(logits, tokens)
    if not res.endswith(" "):
        tokens.extend(model.encode(' ')[0].tolist())
    quotations = []
    if prompt.count('"') >= 2:
        quotations = re.findall(r'"([^"]+)"', prompt)
    elif prompt.count("'") >= 2:
        quotations = re.findall(r"'([^']+)'", prompt)
    if quotations and current_parameter in ("source_string"):
        for q in quotations:
            if q not in res:
                tokens.extend(model.encode(q)[0].tolist())
    elif current_parameter in ("regex", "replacement"):
        return logits
    else:
        for i in prompt.split():
            if i not in res:
                tokens.extend(model.encode(i)[0].tolist())
    return get_masked_logits(logits, tokens)


def value_constrainer(
    function: FunctionDefinition,
    logits: list[float],
    prompt: str,
    output: str,
) -> Any:
    current_parameter = get_current_parameter(output, function)
    if current_parameter is None:
        return logits
    param_type = function.parameters[current_parameter].type
    if param_type == "number":
        return ints_constrained(logits, prompt, output, function)
    if param_type == "float":
        return floats_constrained(logits, prompt)
    if param_type == "bool":
        return bool_constrained(logits)
    if param_type == "string":
        return string_constrained(logits, prompt, output, function)
    return logits
