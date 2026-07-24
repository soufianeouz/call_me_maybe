*This project has been created as part of the 42 curriculum by selouizg.*

## Description

`call me maybe` is a function calling tool that translates natural language prompts into structured function calls using constrained decoding with a small LLM (Qwen3-0.6B).

Given a prompt like "What is the sum of 2 and 3?", the program outputs:
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

## Instructions

### Installation
```bash
make install
```

### Run
```bash
make run
```

### Custom paths
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/tests.json --output data/output/results.json
```

### Lint
```bash
make lint
```

### Clean
```bash
make clean
```

## Algorithm explanation

The program uses **constrained decoding** to guarantee valid structured output :

1. The LLM receives the prompt and available functions
2. At each generation step, invalid tokens are set to `-inf`
3. Only tokens that continue a valid function name are allowed
4. For arguments, tokens are constrained by type (number/string/boolean)
5. The process repeats token by token until generation is complete

## Design decisions

- Used `Qwen/Qwen3-0.6B` as the default model
- Pydantic for input/output validation
- Separated logic into `function_selector.py` and `constrained_decoder.py`
- Context accumulation for multi-parameter extraction
- Stop tokens to signal end of value generation

## Performance analysis

- Function selection accuracy : ~90%+
- Number extraction : works for integers and floats
- String extraction : works for single words and sentences
- Processing time : ~2-5 minutes for full test suite on CPU

## Challenges faced

- Small model (0.6B) unreliable without constrained decoding
- Token boundaries don't always match word boundaries
- Stop condition for number generation required careful handling
- Multi-parameter extraction needed context accumulation

## Testing strategy

- Tested with all provided prompts in `tests.json`
- Tested edge cases : large numbers, missing values, complex strings
- Verified output JSON is valid and schema-compliant
- Tested error handling : missing files, invalid JSON

## Resources

- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)
- [Pydantic documentation](https://docs.pydantic.dev)
- [uv documentation](https://docs.astral.sh/uv)
- [Qwen3 model](https://huggingface.co/Qwen/Qwen3-0.6B)

AI was used to help understand concepts like constrained decoding, tokenization, and logits. AI also helped debug specific issues in the generation loop.