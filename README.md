*This project has been created as part of the 42 curriculum by ckruk*

# Call Me Maybe

## Description

**Call Me Maybe** is a project focused on function calling in Large Language Models (LLMs). This project implements a custom constrained decoding pipeline. Instead of relying solely on prompting, the system intercepts the generation process token-by-token. By analyzing the model's logits against a vocabulary file and function definitions, invalid tokens are masked out (set to negative infinity). This guarantees JSON output transforming natural language requests into precise, executable function calls.

## Instructions

### Algorithm explanation

To force the small Qwen/Qwen3-0.6B model to generate valid JSON structures matching a strict schema, the program implements a token-by-token constrained decoding engine. Instead of relying on prompt engineering alone, the algorithm intercepts the generation process at each step:

1. State-Machine Constrainer (constrainer):
The program checks the current generation state character-by-character using output_state. It determines whether the model expects opening/closing brackets, object keys, colons, commas, or parameter values.

2. Type-Specific Value Constraining:
When a parameter value is expected, specialized filtering functions restrict valid choices based on the Pydantic schema:

- Integers (ints_constrained): Extracts digits and numerical substrings  from the input prompt and allows tokens for those numbers, plus } and ,.
- Floats (floats_constrained): Allows numbers, decimal points (.), negative signs (-), spaces, and commas.
- Booleans (bool_constrained): Restricts tokens strictly to true or false.
- Strings (string_constrained): Handles quote characters and matches literal words, substrings, or text chunks found inside quotes from the original prompt.

3. Logits Masking (get_masked_loggits):
For each token generation step, the engine creates an array filled with -np.inf and injects original logit values only for allowed_tokens. This forces np.argmax(logits) to pick exclusively from valid next tokens.


### Design decisions

- Pydantic Validation: Used BaseModel for parsing and validating functions_definition.json to keep internal data structures strict and clean.
- Modular Constraint Handlers: Separated constraint logic into distinct functions per data type (ints_constrained, string_constrained, etc.) to keep the codebase maintainable.
- Dynamic Key Tracing: Used helper functions (left_parameters, nbr_left_parameters, get_current_parameter) to track which parameters have been generated and prevent duplicate keys.
- Error Handling: Wrapped generation and parsing loops in try-except blocks (catching json.JSONDecodeError) and incorporated a bracket_validator stack to verify JSON syntax integrity.


### Performance analysis

Accuracy: Achieves accuracy on the level required by subject by pulling exact values directly from the prompt context and aligning them with schema definitions.

Speed: Targeted token masking avoids heavy regex post-processing, allowing the program to process prompts quickly and well within the 5-minute requirement.

Reliability: Guarantees syntactically valid JSON output on every run because invalid tokens are blocked at the logit level before decoding.

### Challenges faced

Model was trying to answer the question asked in the prompt, insted of pulling parameters from it. I solved it by giving it acces to use only tokens appearing in the prompt

### Testing strategy

Test was based on modifying prompts to make it more tricky and difficult

• Example usage: Provide clear examples of running your program

### Compilation

```bash
make
```
### Execution / Example usage

```bash
make run
```

### Available Make Commands

```bash
make install      # Synchronizes project dependencies using uv (uv sync)
make run          # Runs the main application module via uv
make debug        # Runs the application inside the Python debugger (pdb)
make clean        # Cleans up cache directories (__pycache__, .mypy_cache, .pytest_cache)
make lint         # Runs code style checks (flake8) and type checking (mypy with custom flags)
make lint-strict  # Runs strict linting and type checking (flake8 and mypy --strict)
```
## Resources

### References

- **Understading the concept**: https://medium.com/data-science-collective/what-is-function-calling-569c0b709062
- **Function calling**: https://www.geeksforgeeks.org/artificial-intelligence/function-calling-in-llms/
- **Understanding pydantic**: https://www.geeksforgeeks.org/python/introduction-to-python-pydantic-library/
- **Understanding LLM**: https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f

### AI Usage

**AI was used for:**
- **Interpretation**: Help to interpretate instructions from the subject
- **Kick-off**: Help with understanding what is required in the project and what is it about to be able to start doing it
- **Explanation**: Guidance with new concepts without giving direct solutions
- **Testing**: Giving examples of cases to test the program
