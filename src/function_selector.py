import json
import numpy as np

def function_selector(prompt, Available_functions, LLM_Model):
    
    functions_str = ""
    
    for item in Available_functions:
        functions_str = functions_str + item["name"] + ": " + item["description"] + "\n"
    
    message = (
        f"You are a function selector.\n"
        f"Given this prompt: '{prompt}'\n"
        f"Choose the right function from this list:\n"
        f"{functions_str}\n"
         f"Read the description carefully before choosing.\n"
        "Answer with only the function name, nothing else"
    )
    
    ids = LLM_Model.encode(message)
    path = LLM_Model.get_path_to_vocab_file()
    with open(path, "r") as f:
        vocabulary = json.load(f)
    
    reversed_vocab = {v:k for k, v in vocabulary.items()}

    
    names_functions = []
    for item in Available_functions:
        names_functions.append(item["name"])

            
    input_ids = list(ids[0])
    current_text = ""
    while current_text not in names_functions:
        logist = LLM_Model.get_logits_from_input_ids(input_ids)
        valid_tokens_id = []
        for word, token_id in vocabulary.items():
             for x in names_functions:
                 if x.startswith(current_text + word):
                    valid_tokens_id.append(token_id)
                    break
        logist = np.array(logist)
        new_logist = logist.copy()
        new_logist[:] = -np.inf
        for token in valid_tokens_id:
            new_logist[token] = logist[token]
        
        next_token_id = int(np.argmax(new_logist)) #np.argmax → finds the index of the biggest value directly.
        input_ids.append(next_token_id)
        
        current_text += reversed_vocab[next_token_id]
    return current_text