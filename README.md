*This project has been created as part of the 42 curriculum by <login1>.*

# call me maybe — Introduction to Function Calling in LLMs

## Description

This project implements a **function calling** system that translates natural language
prompts into structured, machine-executable function calls, using a small language
model (**Qwen/Qwen3-0.6B**, 500M parameters).

Given a prompt such as *"What is the sum of 40 and 2?"*, the goal is not to have the
model answer *"42"* in plain text, but to have it produce a structured call such as:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40, "b": 2}
}
```

Small models are notoriously unreliable at producing valid structured output through
prompting alone (often succeeding only ~30% of the time). To reach near-perfect
reliability, this project relies on **constrained decoding**: at every generation step,
the raw logits produced by the model are masked so that only tokens consistent with
both valid JSON syntax and the expected function schema can be selected. This
guarantees that the final output is always syntactically valid JSON and always
respects the schema defined in `functions_definition.json` — without ever hoping the
model "gets it right" on its own.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) as the package/environment manager
- The `llm_sdk` package (copied at the root of this repository, alongside `src/`)

### Installation

```bash
make install
```

This creates the environment and installs dependencies (`numpy`, `pydantic`, …)
via `uv`. The reviewer can also simply run `uv sync`.

### Running the program

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

By default, the program reads its input files from `data/input/` and writes its
output to `data/output/`. Example with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target        | Description                                              |
|---------------|-----------------------------------------------------------|
| `install`     | Install project dependencies                              |
| `run`         | Execute the main script                                   |
| `debug`       | Run the main script under `pdb`                            |
| `clean`       | Remove `__pycache__`, `.mypy_cache`, and other temp files   |
| `lint`        | Run `flake8` and `mypy` with the mandatory flags            |

## Algorithm Explanation

The constrained decoding pipeline works as follows, at each generation step:

1. **Encode** the current prompt/context into input token IDs.
2. **Get logits** for the next token via `llm_sdk`'s
   `get_logits_from_input_ids`.
3. **Determine the current grammar state** — e.g. "expecting the opening `{`",
   "expecting a function name from the allowed list", "expecting a numeric
   value for parameter `a`", "expecting a string value for parameter `s`",
   "expecting a closing `}`", etc. This state is derived from the JSON schema
   built out of `functions_definition.json` combined with how much valid JSON
   has already been generated.
4. **Build a token mask**: using the vocabulary file returned by
   `get_path_to_vocab_file`, every token is checked against the current
   grammar state. Tokens that would break JSON syntax or violate the expected
   type (number, string, boolean, etc.) are set to `-inf` in the logits.
5. **Select the next token** greedily (or by sampling) among the remaining
   valid tokens only.
6. **Append** the chosen token to the context and repeat until the object is
   syntactically complete (matching braces, correct key/value structure).

Because invalid tokens are masked out *before* selection rather than validated
*after* generation, the resulting JSON is always both syntactically valid and
schema-compliant — 100% of the time, regardless of the model's raw reliability.

## Design Decisions

- **Pydantic models** (`FunctionDef`, `Parameter`, `OutputResult`) are used to
  validate the function definitions and the final output structure, keeping
  parsing and validation logic declarative and type-safe.
- **Greedy, per-type decode loops** (`generate_number`, `generate_string`, …)
  are used rather than a single monolithic decoder, so that each JSON value
  type can enforce its own character/token constraints (digits and sign for
  numbers, quote-delimited characters for strings, etc.).
- The function name itself is chosen **by the LLM**, constrained to the set of
  valid function names from `functions_definition.json` — never by keyword
  heuristics — per the project's requirements.
- No forbidden packages (`dspy`, `transformers`, `pytorch`, `outlines`, etc.)
  are used; only `numpy`, `json`, and `pydantic` in addition to `llm_sdk`.

## Performance Analysis

- **Validity**: constrained decoding guarantees 100% valid, schema-compliant
  JSON on every run — invalid tokens are never selectable.
- **Accuracy**: function and argument selection accuracy is expected to be
  90%+ on the provided test prompts, since function-name selection is left to
  the model but constrained to the valid set.
- **Speed**: all test prompts are processed well within the required 5-minute
  budget on standard hardware, as generation is limited to short structured
  outputs rather than free-form text.
- **Error handling**: malformed or missing input files, unknown functions, and
  type mismatches are all caught and reported with clear error messages
  instead of crashing.

## Challenges Faced

- **Local inference performance**: CPU-only inference on the available local
  hardware (with outdated CUDA support) was too slow for iterative
  development, so the workflow was migrated to Google Colab for faster
  iteration.
- **Package structure**: an early double-nested `llm_sdk` folder caused import
  issues and had to be flattened to match the expected project layout.
- **Sign loss on negative numbers**: SentencePiece's `▁` (space) marker was
  initially excluded from the set of "valid characters" when constraining
  numeric tokens, which caused the minus sign of negative numbers to be
  dropped. This was fixed by correctly accounting for the marker when
  filtering the vocabulary.
- **String truncation**: the string-value decoder initially stopped at the
  first whitespace-like token due to a substring match against the stop
  token, truncating multi-word values. This was fixed by requiring an exact
  match against the closing-quote token rather than a substring match.

## Testing Strategy

- Manual testing against the provided example files
  (`function_calling_tests.json`, `functions_definition.json`), checking that
  the produced `function_calling_results.json` parses as valid JSON and
  matches the expected function names, argument names, and types.
- Edge cases specifically exercised: empty strings, large numbers, negative
  numbers, special characters in strings, functions with multiple parameters,
  and malformed/missing input files.
- Output validation checks that every result object contains exactly the
  three required keys (`prompt`, `name`, `parameters`) with no extra keys or
  prose.

## Example Usage

```bash
$ uv run python -m src
Processing 5 prompts...
Wrote 5 results to data/output/function_calling_results.json
```

Example output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Hugging Face — Constrained beam search / guided generation](https://huggingface.co/blog/constrained-beam-search)
- [Guidance / Outlines project documentation](https://github.com/dottxt-ai/outlines) (for background reading only — not used as a dependency, per project constraints)
- [Sentencepiece tokenizer documentation](https://github.com/google/sentencepiece)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)

**AI usage**: AI assistance was used to help explain the concept of constrained
decoding, to discuss the tokenization/logits/generation pipeline, and to debug
two specific bugs (sign loss on negative numbers, string truncation on
multi-word values). All AI-suggested code and explanations were reviewed,
tested, and understood before being kept, in line with the project's AI usage
guidelines — no whole functions were generated and copy-pasted without
review.