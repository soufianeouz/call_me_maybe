import json
import numpy as np

def generate_number(prompt, param_name, param_type, function_name,LLM_Model, context):
    message = (
        f"Prompt: '{prompt}'\n"
        f"Extract only the value for '{param_name}' from the prompt.\n"
        f"{context}{param_name}="
    )
    ids = LLM_Model.encode(message)
    
    path = LLM_Model.get_path_to_vocab_file()
    with open(path, "r") as f:
        vocabulary = json.load(f)
    
    reversed_vocab = {v: k for k, v in vocabulary.items()}
    current_value = ""
    input_ids = list(ids[0])
    
    stop_tokens = [" ", "\n", "Ċ", "▁"]
    
    while True:
        logist = LLM_Model.get_logits_from_input_ids(input_ids)
        logist = np.array(logist)
        new_logist_copy = logist.copy()
        new_logist_copy[:] = -np.inf
        
        valid_chars = set("0123456789.-")
        for word, token_id in vocabulary.items():
            # allow number tokens
            if all(c in valid_chars for c in word):
                if current_value == "" and not word[0].isdigit():
                    continue
                new_logist_copy[token_id] = logist[token_id]
            # allow stop tokens only if we already have a number
            if word in stop_tokens and current_value != "":
                new_logist_copy[token_id] = logist[token_id]
        
        # no valid tokens → stop
        if new_logist_copy.max() == -np.inf:
            break
        
        next_token_id = int(np.argmax(new_logist_copy))
        next_token = reversed_vocab[next_token_id]
        
        # model picked stop token → number is complete
        if next_token in stop_tokens:
            break
        
        # safety stop
        if len(current_value) > 15:
            break
        
        try:
            float(current_value + next_token)
            current_value += next_token
            input_ids.append(next_token_id)
        except ValueError:
            break
    
    if current_value == "":
        return 0.0
    return float(current_value)


def generate_string(prompt, param_name, LLM_Model, context):
    
    message = (
        f"Prompt: '{prompt}'\n"
        f"Extract only the value for '{param_name}' from the prompt .\n"
        f"{context}{param_name}="
    )
    ids = LLM_Model.encode(message)
    
    current_string = ""
    input_ids = list(ids[0])
    stop_tokens = [" ", "\n", "Ċ", "▁"]
    
    path = LLM_Model.get_path_to_vocab_file()
    
    with open(path, "r") as f:
        vocabulary = json.load(f)
    
    reverse_vocab = {v : k for k, v in vocabulary.items()}
    
    while True:
        # print("S")
        logist = LLM_Model.get_logits_from_input_ids(input_ids)
        logist = np.array(logist)
        # new_logist_copy = logist.copy()
        next_token_id = int(np.argmax(logist))
        
        if any(stop in reverse_vocab[next_token_id] for stop in stop_tokens):
            break
        current_string += reverse_vocab[next_token_id]
        input_ids.append(next_token_id)
        if len(current_string) > 50:
            break
    if current_string == "":
        return ""
    result = current_string.replace("'", "").strip().lstrip("Ġ") 
    # return current_string
    return result.replace("Ġ", " ")    
        
    

def constrained_decoder(prompt, function, LLM_Model):
    result = {}
    
    context = ""
    for param_name, param_info in function["parameters"].items():
        param_type = param_info["type"]

        # pass context to generate_value
        if param_type == "number":
            value = generate_number(prompt, param_name, param_type, function["name"],LLM_Model, context)
        
        # update context with extracted value
        if param_type == "string":
            value = generate_string(prompt, param_name, LLM_Model, context)
        context += f"{param_name}={value}\n"
        result[param_name] = value
    
    return result 