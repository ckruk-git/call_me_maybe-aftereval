import json
from .parser import parse_args
from .finder import choose_function, find_parameters, functions



if __name__ == "__main__":
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    result = []
    i = 0
    nbr_of_prompts = len(prompts)
    while i < nbr_of_prompts:
        prompt = prompts[i]['prompt']
        parameters = {}
        try:
            name = choose_function(prompt, functions)
            parameters = find_parameters(prompt, name)
            result.append({
            "prompt": prompt,
            "name": name.name,
            "parameters": parameters
            })
        except json.JSONDecodeError:
            print(f"Error on parsing {prompt}")
        i += 1
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)