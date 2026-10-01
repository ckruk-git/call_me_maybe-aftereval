from pydantic import BaseModel
from .parser import parse_args
import json, sys
import numpy as np
import re
from .model import model, FunctionDefinition, get_masked_logits
from .constrainers import constrainer
from .utils import key_complited, nbr_left_parameters, get_current_parameter, bracket_validator


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

def choose_function(prompt: str, functions: list[FunctionDefinition]) -> FunctionDefinition:
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

def parameters_request(prompt: str, function: FunctionDefinition) -> str:
    required = list(function.parameters.keys())
    return (
        "You are an engine to extract parameters from user request.\n"
        f"User request is: {prompt}\n"
        f"Function name is {function.name}\n"
        f"Parameters are: {function.parameters}\n"
        f"List of required keys for which you have to extract values is: {required}.\n"
        "Your task now is to extract values from the request.\n"
        "You have to assign values to the keys and return it as JSON.\n"
        "Return ONLY valid JSON object containing required parameters."
        "Do not add any additional text beside JSON.\n" \
        "Usually required parameters are in the request\n"
        "Sometimes request may ask you to replace parameter with regex\n"
        "In such a case you have to find replacement for regex, which is not present in the request\n"
        "Do not add any additional fields or spaces (before or after JSON)"
        "Return ONLY the required parameter keys with their values as valid JSON"
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
) -> str | None:
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
        return

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
        print("OUTPUT:", repr(output))
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