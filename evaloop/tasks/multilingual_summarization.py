"""Code summarization into a natural language other than English.

This is the second half of the "multilingual" loop (code -> summary in language L -> code).
Everything except the language of the summary is kept identical to the standard loop:
same one-sentence instruction, same opener (translated), same docstring wrapper and the
same retained assert line from CodeSummarizationTask.postprocess. The assert line is code,
so it is language neutral and keeps the function name and signature fixed across cycles.
"""

from typing import Any, Dict, Optional

from .code_summarization import CodeSummarizationTask
from .factory import register_task

# opener: translation of the standard loop's "write a python function to"
# ranges: Unicode blocks of the target script, used to measure language compliance
LANGUAGES: Dict[str, Dict[str, Any]] = {
    "hindi": {
        "name": "Hindi",
        "opener": "एक पायथन फ़ंक्शन लिखें जो",
        "ranges": [(0x0900, 0x097F)],
    },
    "chinese": {
        "name": "Chinese",
        "opener": "编写一个Python函数，用于",
        "ranges": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    },
    "spanish": {
        "name": "Spanish",
        "opener": "escribe una función de python que",
        "ranges": None,
    },
}

DEFAULT_TEMPLATE = (
    "Use one sentence to summarize the following code in {language_name} and start with "
    "'{opener}':\n\n```\n{{code}}\n```\n\n```\n{opener}\n```"
)


def script_ratio(text: str, language: str) -> Optional[float]:
    """Share of alphabetic characters written in the target language's script.

    Returns None for Latin-script languages, where the ratio is not informative.
    """
    ranges = LANGUAGES[language]["ranges"]
    if not ranges:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return hits / len(letters)


@register_task("multilingual_summarization")
class MultilingualSummarizationTask(CodeSummarizationTask):
    """Summarize code in one sentence of a non-English natural language."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__(task_config)
        self.language = task_config.get("summary_language", "hindi").lower()
        if self.language not in LANGUAGES:
            raise ValueError(
                f"Unsupported summary language '{self.language}'. "
                f"Available: {', '.join(sorted(LANGUAGES))}"
            )
        spec = LANGUAGES[self.language]
        self.template = task_config.get("prompt_template") or DEFAULT_TEMPLATE.format(
            language_name=spec["name"], opener=spec["opener"]
        )

    def create_prompt(self, input_text: str) -> str:
        return self.template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        # Some models echo the fenced opener back; drop a surrounding fence if present.
        text = raw_response.strip()
        if text.startswith("```") and text.endswith("```") and len(text) > 6:
            text = text[3:-3].strip()
        return text
