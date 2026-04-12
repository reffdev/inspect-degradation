import pytest
from conftest import make_step

from inspect_degradation.grader.rubric import Rubric


def _rubric(user_template: str) -> Rubric:
    return Rubric(
        version=1,
        name="t",
        description="d",
        system="sys",
        user_template=user_template,
    )


def test_loads_packaged_v1_rubric():
    r = Rubric.from_package("step_grader_v1")
    assert r.version == 1
    assert r.name == "step_grader_v1"
    # The packaged rubric must reference all four placeholders the renderer
    # supplies; if a future edit drops one, this catches it.
    assert {"task_goal", "prior_steps", "step_index", "step"} <= r.placeholders


def test_unknown_placeholder_raises_at_load_time():
    with pytest.raises(ValueError, match="unknown"):
        _rubric("hello {bogus}")


def test_positional_placeholder_rejected():
    with pytest.raises(ValueError, match="named"):
        _rubric("hello {}")


def test_render_only_supplies_referenced_fields():
    # The renderer must not KeyError if the rubric uses a subset of fields.
    r = _rubric("goal: {task_goal}")
    out, _ = r.render_user(
        task_goal="finish",
        step_index=0,
        step=make_step(0),
        prior_steps=(),
    )
    assert out == "goal: finish"


def test_render_formats_prior_steps_and_step():
    r = _rubric("step {step_index}:\n{step}\nprior:\n{prior_steps}")
    out, diagnostics = r.render_user(
        task_goal="goal",
        step_index=1,
        step=make_step(1, action="A1", observation="O1"),
        prior_steps=(make_step(0, action="A0", observation="O0"),),
    )
    assert "step 1:" in out
    assert "ACTION:\nA1" in out
    assert "OBSERVATION:\nO1" in out
    assert "--- step 0 ---" in out
    assert diagnostics.prior_steps_truncated is False
    assert diagnostics.prior_steps_kept == 1
    assert diagnostics.prior_steps_dropped == 0


def test_render_caps_prior_context_to_budget():
    r = _rubric("prior:\n{prior_steps}")
    long_action = "X" * 200
    priors = tuple(
        make_step(i, action=long_action, observation="o") for i in range(10)
    )
    out, diagnostics = r.render_user(
        task_goal="g",
        step_index=10,
        step=make_step(10),
        prior_steps=priors,
        prior_context_char_budget=400,
    )
    assert diagnostics.prior_steps_truncated is True
    assert diagnostics.prior_steps_dropped > 0
    assert diagnostics.prior_steps_kept < 10
    # Most-recent step (index 9) must always be present.
    assert "--- step 9 ---" in out
    # Earliest step (index 0) must have been dropped.
    assert "--- step 0 ---" not in out
    assert "earlier" in out and "omitted" in out


def test_render_no_budget_keeps_all_priors():
    r = _rubric("prior:\n{prior_steps}")
    priors = tuple(make_step(i) for i in range(5))
    _, diagnostics = r.render_user(
        task_goal="g",
        step_index=5,
        step=make_step(5),
        prior_steps=priors,
    )
    assert diagnostics.prior_steps_kept == 5
    assert diagnostics.prior_steps_truncated is False
