import json
from .models import FunctionDef, Prompt


def read_json_files(func_file: str, prom_file: str) -> dict:
    function_data = []
    prompt_data = []

    try:
        with open(func_file, "r") as file:
            function_data = json.load(file)

        with open(prom_file, "r") as file:
            prompt_data = json.load(file)

    except FileNotFoundError as err:
        print("FileNotFoundError, there is", err.strerror)
        exit(1)
    except json.JSONDecodeError as err:
        print("invalid json format,", err)
        exit(1)

    try:
        for f in function_data:
            FunctionDef(
                name=f["name"],
                description=f["description"],
                parameters=f["parameters"],
                returns=f["returns"]
            )

        for p in prompt_data:
            Prompt(
                prompt=p["prompt"]
            )
    except Exception as err:
        print(f"Error: invalid data format → {err}")
        exit(1)

    return {
        "functions": function_data,
        "prompts": prompt_data
    }
