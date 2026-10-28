from .parser import parse_args
import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel
import re


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

args = parse_args()
with open(args.functions_definition, "r") as f1:
    raw_functions = json.load(f1)
path = model.get_path_to_vocab_file()
with open(path, encoding="utf-8") as f:
    vocab = json.load(f)
functions = [
    FunctionDefinition.model_validate(function)
    for function in raw_functions
]


def get_masked_logits(
    logits: list[float],
    allowed_tokens: list[int],
) -> list[float]:
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    return list(mask)


def choose_function(prompt: str,
                    functions: list[FunctionDefinition]
                    ) -> FunctionDefinition:
    big_prompt = "Available functions:\n\n"
    for function in functions:
        big_prompt = big_prompt + function.name + '\n'
        big_prompt = big_prompt + function.description + '\n'
    big_prompt = big_prompt + "User request\n"
    big_prompt = big_prompt + prompt + '\n'
    big_prompt = big_prompt + "Choose the best function name.\n"
    big_prompt = big_prompt + "Answer only with the function name.\n"
    input_ids = model.encode(big_prompt)
    input_ids = input_ids[0].tolist()
    prompt_len = len(input_ids)
    for _ in range(20):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = int(np.argmax(logits))
        input_ids.append(next_token)
        generated = model.decode(input_ids[prompt_len:])
        for function in functions:
            if function.name in generated:
                return function
    return functions[0]


def left_parameters(
    function: FunctionDefinition,
    output: str,
) -> Any:
    for parameter in function.parameters:
        if parameter not in output:
            return parameter
    return '"'


def nbr_left_parameters(
    function: FunctionDefinition,
    output: str,
) -> int:
    nbr = 0
    for parameter in function.parameters:
        if parameter not in output:
            nbr += 1
    return nbr


def get_current_parameter(
    output: str,
    function: FunctionDefinition,
) -> str | None:
    last = None
    for parameter in function.parameters:
        if parameter in output:
            last = parameter
        else:
            return last
    return last


def bracket_validator(s: str) -> bool:
    pairs = {')': '(',
             ']': '[',
             '}': '{'}
    stack = []
    for char in s:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0


def key_complited(output: str, function: FunctionDefinition) -> str:
    key = output.split('"')[-1]
    for parameter in function.parameters:
        if parameter == key:
            return "complited"
    return "continue"


def parameters_request(prompt: str, function: FunctionDefinition) -> str:
    required = list(function.parameters.keys())
    return (
        "You are an engine to extract parameters from user request.\n"
        f"User request is: {prompt}\n"
        f"Function name is {function.name}\n"
        f"Parameters are: {function.parameters}\n"
        f"List of required keys for which you have to"
        f"extract values is: {required}.\n"
        "Your task now is to extract values from the request.\n"
        "You have to assign values to the keys and return it as JSON.\n"
        "Return ONLY valid JSON object containing required parameters."
        "Do not add any additional text beside JSON.\n"
        "Usually required parameters are in the request\n"
        "Sometimes request may ask you to replace parameter with regex\n"
        "In such a case you have to find replacement for regex,"
        "which is not present in the request\n"
        "Do not add any additional fields or spaces (before or after JSON)"
        "Return ONLY the required parameter keys with their"
        "values as valid JSON"
        "Don't return type of the parameter\n"
        "Only key and value\n"
        "Example of expected output:\n"
        'prompt: "Greet donkey"\n'
        'name: "fn_greet"\n'
        'parameters: {"name": "donkey"}\n'
        "Return the JSON now:")


def output_state(
    output: str,
    function: FunctionDefinition,
) -> str:
    if output == "":
        return "start"
    last = output[-1]
    if last == "{" or last == ",":
        return "key_start"
    if last == '"' and output.count('"') == 1:
        return "key_content"
    if output.count('"') % 2 == 1:
        last_occurence = output[output.rfind('"') - 1]
        if last_occurence == " ":
            last_occurence = output[output.rfind('"') - 2]
        if last_occurence == "{" or last_occurence == ",":
            if key_complited(output, function) == "complited":
                return "key_end"
            else:
                return "key_content"
    if last == ":" or output[-2] == ":" or last.isdigit():
        return "value"
    if output.count('"') % 2 == 0:
        i = output.rfind('"') - 1
        while output[i] != '"':
            i -= 1
        if output[i - 1] == "{" or output[i - 1] == ",":
            return "colon"
    if output.count('"') % 2 == 1:
        return "value"
    if output.count('"') % 2 == 0 and output.count('"') % 4 == 0:
        if nbr_left_parameters(function, output) != 0:
            return "coma"
        return "over"
    return "value"


def find_parameters(prompt: str,
                    function: FunctionDefinition,
                    ) -> dict[str, object]:
    output = ""
    parameters_prompt = parameters_request(prompt, function)
    input_ids = model.encode(parameters_prompt)
    input_ids = input_ids[0].tolist()
    json_output = {}
    for _ in range(60):
        logits1 = model.get_logits_from_input_ids(input_ids)
        logits = constrainer(output, logits1, function, prompt)
        next_token = int(np.argmax(logits))
        input_ids.append(next_token)
        output += model.decode([next_token])
        if "}" in output and bracket_validator(output):
            try:
                json_output = json.loads(output)
                return json_output
            except json.JSONDecodeError:
                pass
    try:
        json_output = json.loads(output)
    except json.JSONDecodeError:
        pass
    return json_output


def constrainer(
    output: str,
    logits: list[float],
    function: FunctionDefinition,
    prompt: str,
        ) -> Any:
    state = output_state(output, function)
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
        tokens.append(model.encode(token)[0].tolist()[0])
    for token in prompt_splited:
        tokens.extend(model.encode(token)[0].tolist())
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
    if quotations and current_parameter is not None and current_parameter in ("source_string"):
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
