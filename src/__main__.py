import argparse
from .loader import read_json_files
from .function_selector import function_selector
from .constrained_decoder import constrained_decoder
from llm_sdk import Small_LLM_Model
from .models import OutputResult

import json

parser = argparse.ArgumentParser()

parser.add_argument("--input", default="data/input/tests.json")
parser.add_argument("--output", default="data/output/results.json")
parser.add_argument("--functions_definition",  default="data/input/functions_definition.json")

args = parser.parse_args()
module = Small_LLM_Model()

data = read_json_files(args.functions_definition, args.input)


final_result = []
for prompt in data["prompts"]:
    name_func = function_selector(prompt["prompt"], data["functions"], module)
    for functions in data["functions"]:
        if functions["name"] == name_func:
            valid_function = functions
            break
    param = constrained_decoder(prompt["prompt"], valid_function, module)


    try:
        result = OutputResult(
            prompt=prompt["prompt"],
            name=name_func,
            parameters=param
        )
        final_result.append({"prompt": prompt["prompt"], "name": name_func, "parameters": param})
    except Exception as err:
        print(f"Error: invalid output format → {err}")
        exit(1)

with open(args.output, "w") as f:
    json.dump(final_result, f, indent=4)