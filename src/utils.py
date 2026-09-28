def left_parameters(function, output):
    for parameter in function.parameters:
        print("Parameter:", parameter)
        if parameter not in output:
            return parameter
    return '"'

def nbr_left_parameters(function, output):
    nbr = 0
    for parameter in function.parameters:
        if parameter not in output:
            nbr +=1
    return nbr

def get_current_parameter(output, function):
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

def key_complited(output, function):
    key = output.split('"')[-1]
    for parameter in function.parameters:
        if parameter == key:
            return "complited"
    return "continue"