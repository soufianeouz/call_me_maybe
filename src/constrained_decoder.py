import json
import numpy as np
from typing import Any
import re


def generate_number(
    prompt: str, param_name: str, LLM_Model: Any, context: str, param_type: str
) -> float:
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
    marker_chars = "▁Ġ"  # SentencePiece / BPE word-start markers

    while True:
        logist = LLM_Model.get_logits_from_input_ids(input_ids)
        logist = np.array(logist)
        new_logist_copy = logist.copy()
        new_logist_copy[:] = -np.inf

        valid_chars = set("0123456789.-")
        for word, token_id in vocabulary.items():
            stripped = word.lstrip(marker_chars)
            if stripped and all(c in valid_chars for c in stripped):
                if current_value == "" and not (stripped[0].isdigit() or stripped[0] == "-"):
                    continue

                new_logist_copy[token_id] = logist[token_id]
            if word in stop_tokens and current_value != "":
                new_logist_copy[token_id] = logist[token_id]

        if new_logist_copy.max() == -np.inf:
            break

        next_token_id = int(np.argmax(new_logist_copy))
        next_token = reversed_vocab[next_token_id]

        if next_token in stop_tokens:
            break

        if len(current_value) > 15:
            break

        stripped_token = next_token.lstrip(marker_chars)

        if stripped_token == "-" and current_value == "":
            current_value += stripped_token
            input_ids.append(next_token_id)
            continue

        try:
            float(current_value + stripped_token)
            current_value += stripped_token
            input_ids.append(next_token_id)
        except ValueError:
            break

    if current_value == "" or current_value == "-":
        return 0.0
    if param_type == "integer":
        return int(current_value)
    return float(current_value)



SYMBOL_MAP = {
    "asterisk": "*", "asterisks": "*", "star": "*", "stars": "*",
    "underscore": "_", "underscores": "_",
    "hyphen": "-", "hyphens": "-", "dash": "-", "dashes": "-",
    "hashtag": "#", "hashtags": "#", "hash": "#", "pound": "#",
    "dollar sign": "$", "dollar signs": "$",
    "at sign": "@", "at symbol": "@",
}

def normalize_replacement(value: str) -> str:
    v = value.lower().strip()
    if v in SYMBOL_MAP:
        return SYMBOL_MAP[v]
    if len(set(v)) == 1 and v[0] in "*_-#":  # catches "****", "----", etc.
        return v[0]
    return value  # leave literal words/values (e.g. "X", "blue", "NUMBERS") untouched


def normalize_regex(value: str) -> str:
    v = value.strip()
    v = v.strip("()")  # strip stray wrapping parens the model adds
    # common word->pattern fixes the model tends to drift on
    replacements = {
        "aeiou": "[aeiouAEIOU]",
        "AEIOU": "[aeiouAEIOU]",
        "[0-9]+": r"\d+",
    }
    return replacements.get(v, v)

def generate_string(
    prompt: str, param_name: str, LLM_Model: Any, context: str
) -> str:
    substitute_params = {"source_string", "regex", "replacement"}

    if param_name in substitute_params:
        message = (
            "You are extracting one parameter for the function fn_substitute_string_with_regex.\n"
            "This function has three parameters:\n"
            "- source_string: the original text to search in (copy it exactly, unchanged)\n"
            "- regex: the pattern to search for, using ONLY square brackets for character classes "
            "(e.g. \"[aeiouAEIOU]\") — never wrap it in parentheses\n"
            "- replacement: the single symbol or word used for EACH match — always the same short value, "
            "never repeated or multiplied based on how many matches occur "
            "(e.g. 'asterisks' always means \"*\", never \"**\" or \"****\", no matter how many matches there are)\n"
            "\n"
            "Copy or derive only the value asked for. No explanations. Wrap the value in double quotes.\n"
            "\n"
            "Prompt: \"Replace all digits in 'I have 12 cats and 3 dogs' with X\"\n"
            "Parameter: source_string\n"
            "Value: \"I have 12 cats and 3 dogs\"\n"
            "\n"
            "Prompt: \"Replace all digits in 'I have 12 cats and 3 dogs' with X\"\n"
            "Parameter: regex\n"
            "Value: \"[0-9]+\"\n"
            "\n"
            "Prompt: \"Replace all digits in 'I have 12 cats and 3 dogs' with X\"\n"
            "Parameter: replacement\n"
            "Value: \"X\"\n"
            "\n"
            "Prompt: \"Substitute the word 'red' with 'blue' in 'The red car passed the red house'\"\n"
            "Parameter: source_string\n"
            "Value: \"The red car passed the red house\"\n"
            "\n"
            "Prompt: \"Substitute the word 'red' with 'blue' in 'The red car passed the red house'\"\n"
            "Parameter: regex\n"
            "Value: \"red\"\n"
            "\n"
            "Prompt: \"Substitute the word 'red' with 'blue' in 'The red car passed the red house'\"\n"
            "Parameter: replacement\n"
            "Value: \"blue\"\n"
            "\n"
            "Prompt: \"Replace all vowels in 'Hello world' with asterisks\"\n"
            "Parameter: source_string\n"
            "Value: \"Hello world\"\n"
            "\n"
            "Prompt: \"Replace all vowels in 'Hello world' with asterisks\"\n"
            "Parameter: regex\n"
            "Value: \"[aeiouAEIOU]\"\n"
            "\n"
            "Prompt: \"Replace all vowels in 'Hello world' with asterisks\"\n"
            "Parameter: replacement\n"
            "Value: \"*\"\n"
            "\n"
            "Prompt: \"Replace all consonants in 'banana split' with underscores\"\n"
            "Parameter: source_string\n"
            "Value: \"banana split\"\n"
            "\n"
            "Prompt: \"Replace all consonants in 'banana split' with underscores\"\n"
            "Parameter: regex\n"
            "Value: \"[^aeiouAEIOU ]\"\n"
            "\n"
            "Prompt: \"Replace all consonants in 'banana split' with underscores\"\n"
            "Parameter: replacement\n"
            "Value: \"_\"\n"
            "\n"
            f"Prompt: \"{prompt}\"\n"
            f"Parameter: {param_name}\n"
            f"Value: \""
        )
    else:
        message = (
            "You are extracting a function argument from a request.\n"
            "Copy the value EXACTLY as it appears or is implied in the prompt — "
            "no explanations, no rephrasing, no extra words.\n"
            "Wrap the extracted value in double quotes.\n"
            "\n"
            "Prompt: \"What is the product of 3 and 5?\"\n"
            "Parameter: a\n"
            "Value: \"3\"\n"
            "\n"
            "Prompt: \"Execute SQL query 'SELECT * FROM users' on the production database\"\n"
            "Parameter: query\n"
            "Value: \"SELECT * FROM users\"\n"
            "\n"
            "Prompt: \"Read C:\\\\Users\\\\john\\\\config.ini with latin-1 encoding\"\n"
            "Parameter: path\n"
            "Value: \"C:\\\\Users\\\\john\\\\config.ini\"\n"
            "\n"
            f"Prompt: \"{prompt}\"\n"
            f"Parameter: {param_name}\n"
            f"Value: \""
        )

    ids = LLM_Model.encode(message)

    current_string = ""
    input_ids = list(ids[0])
    stop_tokens = ["\"", "\n", "Ċ"]  # stop on closing quote, not on space

    path = LLM_Model.get_path_to_vocab_file()
    with open(path, "r") as f:
        vocabulary = json.load(f)

    reverse_vocab = {v: k for k, v in vocabulary.items()}

    while True:
        logist = LLM_Model.get_logits_from_input_ids(input_ids)
        logist = np.array(logist)
        next_token_id = int(np.argmax(logist))

        token_str = reverse_vocab[next_token_id]
        # print(token_str)

        if any(stop in token_str for stop in stop_tokens):
            # print(token_str.split('"'))
            if len(token_str.split('"')) > 1:
                current_string += token_str.split('"')[0]

            break

        current_string += token_str
        input_ids.append(next_token_id)

        if len(current_string) > 200:  # raised from 50 — SQL/templates/sentences need room
            break

    if current_string == "":
        return ""
    result = current_string.strip().lstrip("Ġ").replace("Ġ", " ")
    if param_name == "replacement":
        result = normalize_replacement(result)
    elif param_name == "regex":
        result = normalize_regex(result)
    return result

def constrained_decoder(
    prompt: str, function: dict, LLM_Model: Any
) -> dict:
    result = {}
    value: Any = None
    context = ""

    for param_name, param_info in function["parameters"].items():
        param_type = param_info["type"]

        if param_type == "number" or param_type == "integer":
            value = generate_number(
                prompt, param_name, LLM_Model, context, param_type
            )
        if param_type == "string":
            value = generate_string(
                prompt, param_name, LLM_Model, context
            )
        context += f"{param_name}={value}\n"
        result[param_name] = value
    return result