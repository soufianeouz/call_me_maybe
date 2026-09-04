import argparse
import json
from .loader import read_json_files
from .function_selector import function_selector
from .constrained_decoder import constrained_decoder
from llm_sdk import Small_LLM_Model
from .models import OutputResult
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument("--output", default="data/output/function_calling_results.json")

    args = parser.parse_args()
    module = Small_LLM_Model()

    data = read_json_files(args.functions_definition, args.input)


    final_result = []
    for i, prompt in enumerate(data["prompts"]):
        print(f'[{i + 1}/{len(data["prompts"])}] {prompt["prompt"]}')
        name_func = function_selector(
            prompt["prompt"], data["functions"], module
        )
        for functions in data["functions"]:
            if functions["name"] == name_func:
                valid_function = functions
                break
        param = constrained_decoder(
            prompt["prompt"], valid_function, module
        )

        try:
            OutputResult(
                prompt=prompt["prompt"],
                name=name_func,
                parameters=param
            )
            final_result.append({
                "prompt": prompt["prompt"],
                "name": name_func,
                "parameters": param
            })
        except Exception as err:
            print(f"Error: invalid output format → {err}")
            exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(final_result, f, indent=4)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(e)