import json, sys
from llm_sdk import Small_LLM_Model
import numpy as np


model = Small_LLM_Model()
with open("data/input/function_calling_tests.json", "r") as f2:
    prompts = json.load(f2)

with open("data/input/functions_definition.json", "r") as f1:
    functions = json.load(f1)

path = model.get_path_to_vocab_file()
with open(path, encoding="utf-8") as f:
    vocab = json.load(f)

def choose_function(prompt: str, functions: dict) -> str:
    big_prompt = "Available functions:\n\n"
    for function in functions:
        big_prompt = big_prompt + function['name'] + '\n'
        big_prompt = big_prompt + function['description'] + '\n'
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
            if function['name'] in generated:
                return function
    return functions[0]

def ints_constrained(logits):
    allowed = ["}", "-", ",", " ",]
    for i in range(10):
        allowed.append(str(i))
    tokens = []
    for i in allowed:
        tokens.extend(model.encode(i)[0].tolist())
    return get_masked_loggits(logits, tokens)


def floats_constrained(logits, prompt):
    prompt = prompt.split()
    allowed = ["}", ",", "-", ".", " "]
    for i in range(10):
        allowed.append(str(i))
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    for i in prompt:
        tokens.extend(model.encode(i)[0].tolist())    
    return get_masked_loggits(logits, tokens)


def bool_constrained(logits):
    allowed = ["true", "false", "}", "'", " "]
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])

    return get_masked_loggits(logits, tokens)

def string_constrained(logits, prompt):
    prompt = prompt.split()
    allowed = ['"', '}', ',', ' ']
    tokens = []
    for i in allowed:
        tokens.extend(model.encode(i)[0].tolist())
    for i in prompt:
        print(i)
        tokens.extend(model.encode(i)[0].tolist())
    return get_masked_loggits(logits, tokens)
    
    
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


def parameters_request(prompt, function):
    return (
        "You are extracting arguments for a function call.\n"
        "IMPORTANT: Do NOT execute the function.\n"
        "Do NOT calculate anything.\n"
        "Do NOT transform, reverse, replace, substitute, or modify any value.\n"
        "Only extract the arguments that the user provided or requested.\n"
        "Parameters are inside the user request.\n"
        "So the answer will always be words/characters that are in the request\n"
        "You just have to find it in the user prompt\n"
        "So the answer can't be anything that doesn't appear in user request\n"
        "You are not allowed to change form of the parameter\n"
        "It has to be returned the way it appears in the user request\n"
        "Parameter has to appear in the request in the exact, non changed form\n"
        "Return ONLY one JSON object.\n"
        "Use exactly the parameter names defined by the function.\n"
        "Do not invent parameters.\n"
        "Do not add explanations.\n"
        f"User request: {prompt}\n"
        f"Function: {function['name']}\n"
        f"Parameters: {json.dumps(function['parameters'])}\n\n"
        "JSON:"
    )


def get_masked_loggits(logits, allowed_tokens) -> list[float]:
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    return list(mask)

def output_state(output: str):
    if output == "":
        return "start"
    if output == "{":
        return "key_start"
    if output.endswith(":"):
        return "value"
    if output.endswith(","):
        return "key_start"
    if output.endswith('}'):
        return "end"
    if output.endswith('"'):
        before = output[:-1].rstrip()
        if before.endswith('{') or before.endswith(','):
            return "key_content"
        if before.endswith(':'):
            return "value"
        last_segment = output.split(",")[-1]
        if ":" in last_segment:
            after_colon = last_segment.split(":", 1)[1]
        else:
            after_colon = ""
        if after_colon.count('"') >= 2:
            return "value_done"
        return "key_end"
    last_segment = output.split(",")[-1]
    if '"' in last_segment and ':' not in last_segment:
        return "key_content"
    # liczba po ':'
    if ":" in output:
        after_colon = output.split(":", 1)[1].strip()
        if after_colon and after_colon[-1].isdigit():
            return "value_done"
        if after_colon in ("true", "false"):
            return "value_done"
    return "value"

def constrainer(output, logits, function, prompt):
    next_parameter = None
    state = output_state(output)
    if state == "key_end" or state == "value":
        next_parameter = left_parameters(function, output)
        if next_parameter == '"' and output[:-1].rstrip() != '"':
            tokens.append(model.encode('"')[0].tolist()[0])
            return get_masked_loggits(logits, tokens)
    tokens = []
    if state == "start":
        tokens.append(model.encode('{')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "key_start":
        tokens.append(model.encode('"')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "key_content":
        left_par = left_parameters(function, output)
        tokens.append(model.encode(left_par)[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "key_end":
        tokens.append(model.encode(":")[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "value":
        return value_constrainer(function, logits, prompt, next_parameter)
    if state == "value_done":
        if nbr_left_parameters(function, output) == 0:
            tokens.append(model.encode('}')[0].tolist()[0])
        else:
            tokens.append(model.encode(',')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    return logits

def get_current_parameter(output, function):
    for parameter in function["parameters"]:
        if output.endswith(f'"{parameter}"'):
            return parameter
    return None

def left_parameters(function, output):
    for parameter in function["parameters"]:
        if output.endswith(parameter):
            return '"'
        if parameter not in output:
            return parameter
    return '"'

def nbr_left_parameters(function, output):
    nbr = 0
    for parameter in function["parameters"]:
        if parameter not in output:
            nbr +=1
    return nbr

def value_constrainer(function, logits, prompt, current_parameter):
    if current_parameter is None:
        return logits
    param_type = function["parameters"][current_parameter]["type"]
    print("Param_type")
    print(param_type)
    if param_type == "number":
        return ints_constrained(logits)
    if param_type == "float":
        return floats_constrained(logits, prompt)
    if param_type == "bool":
        return bool_constrained(logits)
    if param_type == "string":
        return string_constrained(logits, prompt)
    return logits

def find_parameters(prompt, function) -> json:
    output = ""
    parameters_prompt = parameters_request(prompt, function)
    input_ids = model.encode(parameters_prompt)
    input_ids = input_ids[0].tolist()
    json_output = {}
    for _ in range(20):
        logits1 = model.get_logits_from_input_ids(input_ids)
        logits = constrainer(output, logits1, function, prompt)
        next_token = int(np.argmax(logits))
        input_ids.append(next_token)
        output += model.decode([next_token])
        print(output)
        if output.endswith("}") and bracket_validator(output):
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


result = []
i = 0
nbr_of_prompts = len(prompts)
while i < nbr_of_prompts:
    prompt = prompts[i]['prompt']
    parameters = {}
    try:
        name = choose_function(prompt, functions)
        parameters = find_parameters(prompt, name)
        print(parameters)
    except json.JSONDecodeError:
        print(f"Error on parsing {prompt}")
    result.append({
        "prompt": prompt,
        "name": name['name'],
        "parameters": parameters
        })
    print(parameters)
    i += 1
with open("data/output/function_calling_results.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)
