import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from typing import List

answer_letter_long = """Question: {question}
{choices_text}
Answer with a single letter."""

answer_letter_short = """Question: {question}
{choices_text}
Answer with a single letter."""

answer_letter_base = """Question: {question}

{choices_text}

Answer: The correct answer is """

answer_math = """Question: {question}
The final answer MUST BE put in \\boxed{{}} and no explanation."""

answer_math_qwen = """Question: {question}
ONLY return the final result in LaTeX with no words.
The result MUST be wrapped inside \\boxed{{...}}."""


answer_math_base = """Question: What is $2 + 3$?
Answer: \\boxed{{5}}

Question: What is $10 \\div 2$?
Answer: \\boxed{{5}}

Question: {question}
Answer: \\boxed{{"""

def format_choices(choices: List[str]) -> str:
    return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

def format_choices_base(choices: List[str]) -> str:
    return "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])


