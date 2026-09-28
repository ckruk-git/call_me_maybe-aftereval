import json, sys
from llm_sdk import Small_LLM_Model
import numpy as np
from . import utils


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
    function_name_generated = False
    while not function_name_generated:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = np.argmax(logits)
        input_ids.append(next_token)
        generated = model.decode(input_ids[prompt_len:])
        for function in functions:
            if function['name'] in generated:
                function_name_generated = True
                chosen_function = function
    return chosen_function


def type_analyzer(function):
    params = function['parameters']
    res = [val['type'] for val in params.values()]
    return res


def ints_constrained(logits):
    allowed = ["}", "-", " ", ":", "'", "{",]
    for i in range(10):
        allowed.append(str(i))
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    return get_masked_loggits(logits, tokens)


def floats_constrained(logits):
    allowed = ["}", ",", "-", "."]
    for i in range(10):
        allowed.append(str(i))
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    return get_masked_loggits(logits, tokens)


def bool_constrained(logits):
    allowed = ["True", "False", "}", "parameters", ":", "'", "{",]
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
    return get_masked_loggits(logits, tokens)

def constrain_choser(function, output: str, logits) -> any:
    typ = type_analyzer(function)
    if ((all(i == "number" for i in typ)) and last_symbol(output) == 2):
        return ints_constrained(logits)
    if ((all(i == "bool" for i in typ)) and last_symbol(output) == 2):
        return bool_constrained(logits)
    if ((all(i == "float" for i in typ)) and last_symbol(output) == 2):
        return floats_constrained(logits)
    if (last_symbol(output) == 1):
        return key_constrained(logits, function)
    return None

def key_constrained(logits, function):
    required_keys = list(function["parameters"].keys())
    allowed = ["[", "]", "'", ":", " ", '"', "{", "}"]
    allowed.extend(required_keys)
    tokens = []
    for i in allowed:
        tokens.append(model.encode(i)[0].tolist()[0])
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


def parameters_request(prompt: str, function: dict):
    required_keys = list(function["parameters"].keys())
    print(required_keys)
    print(prompt)
    return (
        "You are engine to extract parameters from user request.\n"
        f"User request is: {prompt}\n"
        f"Function name is {function["name"]}\n"
        #f"List of required keys for which you have to extract values is: {required_keys}"
        "Your task now is to extract values from the request.\n"
        "You have to assign values to the keys and return it as JSON.\n"
        "Return ONLY valid JSON object containing required parameters."
        "Do not add any additional text beside JSON.\n"
        "Do not add also any additional fields or spaces (before or after JSON)"
        "Return ONLY the required parameter keys with their values as valid JSON"
        "Don't return type of the parameter\n"
        "Only key and value from the prompt\n"
        #"Value has to be extracted from the user request\n"
        f"Parameters are: {function["parameters"]}\n"
        "Example of expected format:\n"
        '{"key": value, "key": value}\n'
        "Example 1:"
        'prompt: "Greet donkey"\n'
        'name: "fn_greet"\n'
        'parameters: {"s": "donkey"}\n'
        "Example 2:"
        'prompt: "What is the sum of 242 and 315?"\n'
        'name: "fn_add_numbers"\n'
        'parameters: {"a": 242, "b": 315}\n'
        "Return the JSON now:")


def get_masked_loggits(logits, allowed_tokens) -> list[int]:
    copy = np.array(logits, copy=True)
    mask = np.full(copy.shape, -np.inf)

    mask[allowed_tokens] = copy[allowed_tokens]
    mask_list = list(mask)
    return mask_list

def last_symbol(output: str) -> int:
    if not output:
        return 1
    if output[-1:] == ":":
        return 2
    if output[-1:] == "{" or output[-1:] == ",":
        return 1
    return 0


def find_parameters(prompt, function) -> json:
    output = ""
    parameters_prompt = parameters_request(prompt, function)
    input_ids = model.encode(parameters_prompt)
    input_ids = input_ids[0].tolist()
    json_output = {}
    for _ in range(60):
        logits = model.get_logits_from_input_ids(input_ids)  #to zwraca 3,67, 8,15 itd
        try:
            if constrain_choser(function, output, logits) is not None:
                logits = constrain_choser(function, output, logits)
        except json.JSONDecodeError:
            pass
        next_token = np.argmax(logits)
        input_ids.append(next_token)
        output += model.decode(next_token)
        if '}' in output and bracket_validator(output):
            try:
                print(repr(output))
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
    parameters = []
    try:
        name = choose_function(prompt, functions)
        parameters = find_parameters(prompt, name)
    except json.JSONDecodeError:
        print(f"Error on parsing {prompt}")
    result.append({
        "prompt": prompt,
        "name": name['name'],
        "parameters": parameters
        })
    i += 1
with open("data/output/function_calling_results.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)



"""print(prompts[0]['prompt'])
print(repr(prompts[0]['prompt']))
test1 = choose_function(prompts[0]['prompt'], functions)
print(test1)
test2 = find_parameters(prompts[0]['prompt'], test1)
print(test2)"""



"""for token in ["{", "}", '"', ":", ",", "0", "1", "2", "3"]:
    print(token, vocab.get(token))

for prompt in prompts:
    print(prompt["prompt"])
    a = choose_function(prompt["prompt"], functions)
    print(a["name"])
    b = find_parameters(prompt["prompt"], a)
    print(b)



state = 1
for state in range(5):


input_ids = model.encode("Generate a number")
input_ids = input_ids[0].tolist()
logits = model.get_logits_from_input_ids(input_ids)
allowed = set(range(15, 25))
for token in allowed:
    masked_logits[token] = logits[token]

next_token = np.argmax(masked_logits)

print(next_token)
print(model.decode([next_token]))
"""
