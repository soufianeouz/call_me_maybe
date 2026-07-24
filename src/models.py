from pydantic import BaseModel
from typing import Any


class Parameter(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    prompt: str


class OutputResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
