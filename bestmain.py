import json, sys
from llm_sdk import Small_LLM_Model
import numpy as np
import re

model = Small_LLM_Model()
with open("data/input/function_calling_tests.json", "r") as f2:
    prompts = json.load(f2)

with open("data/input/functions_definition.json", "r") as f1:
    functions = json.load(f1)

path = model.get_path_to_vocab_file()
with open(path, encoding="utf-8") as f:
    vocab = json.load(f)

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


def get_masked_loggits(logits, allowed_tokens) -> list[float]:
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    return list(mask)


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


"""def parameters_request(prompt, function):
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
        "Do not add explanations, neither new line at the end.\n"
        f"User request: {prompt}\n"
        f"Function: {function['name']}\n"
        f"Parameters: {json.dumps(function['parameters'])}\n\n"
        "JSON:"
    )"""

def parameters_request(prompt: str, function: dict):
    required = list(function["parameters"].keys())
    hints = ""
    if "source_string" in required:
        hints += "- source_string: the FULL string being operated on (longest quoted text)\n"
    if "regex" in required:
        hints += "- regex: the pattern to search for\n"
    if "replacement" in required:
        hints += "- replacement: the text to replace with\n"
    print(prompt)
    return (
        "You are engine to extract parameters from user request.\n"
        f"User request is: {prompt}\n"
        f"Function name is {function['name']}\n"
        "Your task now is to extract values from the request.\n"
        "You have to assign values to the keys and return it as JSON.\n"
        "Return ONLY valid JSON object containing required parameters."
        "Do not add any additional text beside JSON.\n"
        "Do not add also any additional fields or spaces (before or after JSON)"
        "Return required parameter keys with their values as valid JSON"
        f"Parameters are: {required}\n"
        f"{hints}\n"
        "Return the JSON now:")


def ints_constrained(logits, prompt, output):
    tokens = []
    res = output.split(":", 1)[1]
    print("res", res)
    prompt = prompt.split()
    allowed = ['}', ',']
    for i in prompt:
        numbers = re.findall(r'\d+', i)
        if numbers:
            allowed.extend(numbers)
    print("Allowed", allowed)
    for i in allowed:
        if i not in res:
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

def string_constrained(logits, prompt, output, function):
    required = list(function["parameters"].keys())
    if "regex" in required or "replacement":
        pass
    tokens = []
    res = output.split(":", 1)[1]
    tokens.extend(model.encode('"')[0].tolist())
    if len(res) == 0:
        return get_masked_loggits(logits, tokens)
    if not res.endswith(" "):
        tokens.extend(model.encode(' ')[0].tolist())
    quotations = []
    if prompt.count('"') >= 2:
        quotations = re.findall(r'"([^"]+)"', prompt)
    elif prompt.count("'") >= 2:
        quotations = re.findall(r"'([^']+)'", prompt)
    if quotations:
        next_quote = None
        for q in quotations:
            if q not in res:
                next_quote = q
                break
        if next_quote is not None:
            tokens.extend(model.encode(next_quote)[0].tolist())
    else:
        for i in prompt.split():
            if i not in res:
                tokens.extend(model.encode(i)[0].tolist())
    return get_masked_loggits(logits, tokens)

def key_complited(output, function):
    key = output.split('"')[-1]
    for parameter in function["parameters"]:
        if parameter == key:
            return "complited"
    return "continue"

def output_state(output: str, function):
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

def constrainer(output, logits, function, prompt):
    state = output_state(output, function)
    print("STATE:", state)
    tokens = []  
    if state == "start":
        tokens.append(model.encode('{')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "key_start" or state == "key_end" or state == "value_end":
        tokens.append(model.encode('"')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "key_content":
        left_par = left_parameters(function, output)
        tokens.extend(model.encode(left_par)[0].tolist())
        return get_masked_loggits(logits, tokens)
    if state == "colon":
        tokens.append(model.encode(':')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "coma":
        tokens.append(model.encode(',')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "over":
        tokens.append(model.encode('}')[0].tolist()[0])
        return get_masked_loggits(logits, tokens)
    if state == "value":
        return value_constrainer(function, logits, prompt, output)

def get_current_parameter(output, function):
    last = None
    for parameter in function["parameters"]:
        if parameter in output:
            last = parameter
        else:
            return last
    return last

def left_parameters(function, output):
    for parameter in function["parameters"]:
        print("Parameter:", parameter)
        if parameter not in output:
            return parameter
    return '"'

def nbr_left_parameters(function, output):
    nbr = 0
    for parameter in function["parameters"]:
        if parameter not in output:
            nbr +=1
    return nbr

def value_constrainer(function, logits, prompt, output):
    current_parameter = get_current_parameter(output, function)
    if current_parameter is None:
        return logits
    param_type = function["parameters"][current_parameter]["type"]
    if param_type == "number":
        return ints_constrained(logits, prompt, output)
    if param_type == "float":
        return floats_constrained(logits, prompt)
    if param_type == "bool":
        return bool_constrained(logits)
    if param_type == "string":
        return string_constrained(logits, prompt, output, function)
    return logits

def find_parameters(prompt, function) -> json:
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


result = []
i = 8
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
