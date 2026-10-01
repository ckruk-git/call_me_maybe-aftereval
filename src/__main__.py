import json, sys
from llm_sdk import Small_LLM_Model
import numpy as np
import re
from .parser import parse_args
from . import utils
from .constrainers import constrainer
from .model import model
from .finder import choose_function, find_parameters, functions

args = parse_args()

with open(args.input, "r") as f2:
    prompts = json.load(f2)

if __name__ == "__main__":
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
            "name": name.name,
            "parameters": parameters
            })
        print(prompt)
        print(parameters)
        i += 1
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
