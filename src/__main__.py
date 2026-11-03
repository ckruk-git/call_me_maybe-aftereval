import json, sys, os
from .parser import parse_args
from .finder import choose_function, find_parameters, load_functions


if __name__ == "__main__":
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        with open(args.input, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    result = []
    i = 0
    nbr_of_prompts = len(prompts)
    while i < nbr_of_prompts:
        try:
            prompt = prompts[i]['prompt']
        except KeyError:
            print(f"Missing 'prompt' key in entry {i}")
            i += 1
            continue
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
    try:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
    except FileNotFoundError:
        print(f"File not found")