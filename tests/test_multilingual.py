"""Tests for the multilingual summarization loop."""

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.tasks import MultilingualSummarizationTask, TaskFactory
from evaloop.tasks.multilingual_summarization import LANGUAGES, script_ratio

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT

HINDI = {"type": "multilingual_summarization", "summary_language": "hindi"}


def test_registered():
    task = TaskFactory().create_task(HINDI)
    assert isinstance(task, MultilingualSummarizationTask)
    assert task.produces_testable_code is False


def test_prompt_mirrors_standard_loop():
    prompt = MultilingualSummarizationTask(HINDI).create_prompt("def f(x): return x")
    assert prompt.startswith("Use one sentence to summarize the following code in Hindi")
    assert "def f(x): return x" in prompt
    assert prompt.count(LANGUAGES["hindi"]["opener"]) == 2


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        MultilingualSummarizationTask({"summary_language": "klingon"})


def test_postprocess_keeps_assert_line():
    task = MultilingualSummarizationTask(HINDI)
    summary = "एक पायथन फ़ंक्शन लिखें जो दो सूचियों के साझा तत्व लौटाए।"
    out = task.postprocess(task.extract_output(summary), {"cycle_input": MBPP_STYLE_PROMPT})
    assert out == f'"""\n{summary}\n{ASSERT_LINE}\n"""\n'
    # the wrapped summary is a valid cycle input for the next postprocess
    assert task.postprocess("x", {"cycle_input": out}).splitlines()[-2] == ASSERT_LINE


def test_extract_output_strips_fence():
    task = MultilingualSummarizationTask(HINDI)
    assert task.extract_output("```\nएक पायथन\n```") == "एक पायथन"


def test_script_ratio():
    assert script_ratio("एक पायथन फ़ंक्शन", "hindi") == 1.0
    assert script_ratio("write a python function", "hindi") == 0.0
    assert script_ratio("anything", "spanish") is None


def test_config_wiring(tmp_path):
    cfg = EvaluationConfig(model="gemini-2.5-flash", loop="multilingual", output_dir=str(tmp_path))
    gen, summ = cfg.get_task_configs(cfg.loop_task_type())
    assert gen["type"] == "code_generation"
    assert summ == {**HINDI, "strip_code": False}
    assert cfg.get_model_config()["type"] == "gemini"
    std = EvaluationConfig(model="gemini-2.5-flash", output_dir=str(tmp_path))
    assert std.get_task_configs(std.loop_task_type())[0] == gen


def test_local_backend_config(tmp_path):
    cfg = EvaluationConfig(model="Qwen/Qwen2.5-Coder-3B-Instruct", backend="local", engine="hf",
                           output_dir=str(tmp_path))
    mc = cfg.get_model_config()
    assert mc["type"] == "local_chat"
    assert mc["path"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert mc["engine"] == "hf"


def test_strip_code_blocks():
    from evaloop.tasks.code_summarization import strip_code_blocks
    leaky = "एक पायथन फ़ंक्शन लिखें जो सूची का योग दे।\n\n```python\ndef f(x):\n    return sum(x)\n```\nउदाहरण:\n```python\nprint(f([1]))\n```"
    assert strip_code_blocks(leaky) == "एक पायथन फ़ंक्शन लिखें जो सूची का योग दे।\n\nउदाहरण:"
    assert strip_code_blocks("no fences here") == "no fences here"
    assert strip_code_blocks("```\nwrite a python function to add\n```") == "write a python function to add"
    assert strip_code_blocks("text\n```python\ndef f(): pass") == "text"


def test_strip_code_option_wiring(tmp_path):
    cfg = EvaluationConfig(model="gemini-2.5-flash", loop="multilingual", strip_summary_code=True, output_dir=str(tmp_path))
    summ = cfg.get_task_configs(cfg.loop_task_type())[1]
    assert summ["strip_code"] is True
    task = TaskFactory().create_task(summ)
    out = task.postprocess("सारांश\n```python\ndef f(): pass\n```", {"cycle_input": MBPP_STYLE_PROMPT})
    assert out == f'"""\nसारांश\n{ASSERT_LINE}\n"""\n'
    std = EvaluationConfig(model="gemini-2.5-flash", output_dir=str(tmp_path))
    assert std.get_task_configs(std.loop_task_type())[1]["strip_code"] is False
