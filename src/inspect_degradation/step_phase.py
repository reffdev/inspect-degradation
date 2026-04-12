"""Classify agent steps as exploration vs action.

Deterministic, zero-cost classification from the step's action text.
No grader call needed — this is a computed column, not a rubric
dimension.

Exploration steps (reading, searching, navigating) are fundamentally
less consequential than action steps (editing, running, submitting).
A wrong exploration step (searched the wrong directory) is recoverable;
a wrong action step (edited the wrong file) may not be. Using this as
a covariate in the mixed-effects model controls for the confound that
later steps in a trace tend to be actions (edits, submits) while
earlier steps tend to be exploration (reads, searches).

Supports multiple agent frameworks:
- Auto-SWE: ``[tool_call: toolName] {json}``
- OpenHands: ``[bash] command`` / ``[str_replace_editor] {json}``
- SWE-agent / terminus (XML): ``<execute_bash><command>...</command></execute_bash>``
- Plain shell commands (fallback)
"""

from __future__ import annotations

import re

# ── Auto-SWE structured tool calls ────────────────────────────────

_TOOL_CALL_RE = re.compile(r"\[tool_call:\s*(\w+)\]")

_AUTOSWE_EXPLORE_TOOLS = frozenset({
    "readfile", "searchfiles", "listdirectory", "readrelevantfiles",
    "lookupdocs", "gitdiff", "gitstatus", "getrelatedstories", "todo",
})

_AUTOSWE_ACT_TOOLS = frozenset({
    "runcommand", "replaceinfile", "writefile", "appendtofile",
    "deletefile", "checkbuild", "checktests", "checklint",
    "checkpackage", "submitverdict", "submitresult", "submittestresult",
})

# ── OpenHands bracket-style tool calls ────────────────────────────

# Matches [tool_name] at start of a line, capturing tool + optional subcommand.
_OH_TOOL_RE = re.compile(r"^\[(\w+(?:_\w+)*)\]\s*(\w+)?", re.MULTILINE)

_OH_PURE_ACT_TOOLS = frozenset({
    "submit", "finish", "vim", "edit",
})

_OH_PURE_EXPLORE_TOOLS = frozenset({
    "browser",
})

# str_replace_editor subcommands — the tool name alone is ambiguous.
_OH_EDITOR_EXPLORE_SUBS = frozenset({"view"})
_OH_EDITOR_ACT_SUBS = frozenset({
    "str_replace", "create", "insert", "undo_edit",
})

# ── XML tool blocks (SWE-agent, terminus) ─────────────────────────

_XML_COMMAND_RE = re.compile(
    r"<execute_(?:bash|ipython)>\s*(?:<command>)?\s*(.*?)\s*(?:</command>)?\s*</execute_",
    re.DOTALL | re.IGNORECASE,
)

# ── Shell command patterns (used for XML content and plain text) ──

_SHELL_EXPLORE_RE = re.compile(
    r"(?:^|\s|/|;|&&|\|\|)"  # preceded by boundary
    r"(find\s|grep\s|rg\s|ag\s|"
    r"cat\s|head\s|tail\s|less\s|more\s|"
    r"ls(?:\s|$)|dir\s|pwd|cd\s|tree(?:\s|$)|"
    r"wc\s|file\s|stat\s|"
    r"git\s+(?:log|diff|show|blame|status)\b)",
    re.IGNORECASE | re.MULTILINE,
)

_SHELL_ACT_RE = re.compile(
    r"(?:^|\s|/|;|&&|\|\|)"  # preceded by boundary
    r"(sed\s|awk\s|patch\s|"
    r"git\s+(?:add|commit|push|checkout|rebase|merge|cherry-pick)\b|"
    r"mv\s|cp\s|rm\s|mkdir\s|"
    r"pip\s+install|python(?:3)?\s|pytest\s|"
    r"make(?:\s|$)|npm\s|cargo\s|"
    r"kill\s|"
    r"tee\s|chmod\s|chown\s)",
    re.IGNORECASE | re.MULTILINE,
)

# SWE-agent specific commands
_SWEAGENT_EXPLORE_RE = re.compile(
    r"\b(find_file|search_file|search_dir|open\s|goto\s|scroll_up|scroll_down)\b",
    re.IGNORECASE,
)

_SWEAGENT_ACT_RE = re.compile(
    r"\b(edit\s|create\s|submit\b)\b",
    re.IGNORECASE,
)


def _classify_shell(text: str) -> str | None:
    """Classify shell command text. Returns None if ambiguous."""
    has_act = bool(_SHELL_ACT_RE.search(text)) or bool(_SWEAGENT_ACT_RE.search(text))
    has_explore = bool(_SHELL_EXPLORE_RE.search(text)) or bool(_SWEAGENT_EXPLORE_RE.search(text))

    if has_act and not has_explore:
        return "act"
    if has_explore and not has_act:
        return "explore"
    if has_act and has_explore:
        return "act"  # mixed → act wins
    return None


def classify_step_phase(action: str) -> str:
    """Classify a step as ``'explore'`` or ``'act'`` from its action text.

    Uses a layered strategy:

    1. **Structured tool calls** — if the action contains
       ``[tool_call: X]`` (Auto-SWE) or ``[toolname]`` (OpenHands),
       classify based on the tool name. This avoids false matches on
       code content inside tool arguments.
    2. **XML tool blocks** — extract the command from
       ``<execute_bash>`` blocks and classify the command only.
    3. **Plain text fallback** — match shell command patterns.

    If nothing matches, defaults to ``'act'``.
    """
    # ── 1. Auto-SWE: [tool_call: toolName] ──────────────────────
    tool_calls = _TOOL_CALL_RE.findall(action)
    if tool_calls:
        has_act = any(tc.lower() in _AUTOSWE_ACT_TOOLS for tc in tool_calls)
        has_explore = any(tc.lower() in _AUTOSWE_EXPLORE_TOOLS for tc in tool_calls)
        if has_act:
            return "act"
        if has_explore:
            return "explore"
        return "act"  # unknown tool → act

    # ── 2. OpenHands: [tool] subcommand ───────────────────────────
    oh_matches = _OH_TOOL_RE.findall(action)
    if oh_matches:
        # Each match is (tool_name, subcommand_or_empty).
        has_act = False
        has_explore = False

        for tool, sub in oh_matches:
            tl = tool.lower()
            sl = sub.lower() if sub else ""

            if tl in _OH_PURE_ACT_TOOLS:
                has_act = True
            elif tl in _OH_PURE_EXPLORE_TOOLS:
                has_explore = True
            elif tl == "str_replace_editor":
                if sl in _OH_EDITOR_EXPLORE_SUBS:
                    has_explore = True
                elif sl in _OH_EDITOR_ACT_SUBS:
                    has_act = True
                else:
                    has_act = True  # unknown subcommand → act
            elif tl in ("bash", "execute_bash"):
                # Classify the command content after [bash].
                bash_cmds = re.findall(
                    r"^\[" + re.escape(tool) + r"\]\s*(.+?)$",
                    action, re.MULTILINE,
                )
                cmd_text = " ".join(bash_cmds)
                shell = _classify_shell(cmd_text)
                if shell == "act":
                    has_act = True
                elif shell == "explore":
                    has_explore = True
                else:
                    has_act = True  # unknown bash command → act

        if has_act:
            return "act"
        if has_explore:
            return "explore"
        return "act"

    # ── 3. XML blocks: <execute_bash><command>...</command> ─────
    xml_cmds = _XML_COMMAND_RE.findall(action)
    if xml_cmds:
        cmd_text = " ".join(xml_cmds)
        result = _classify_shell(cmd_text)
        if result:
            return result
        return "act"

    # ── 4. Plain text fallback ──────────────────────────────────
    result = _classify_shell(action)
    if result:
        return result
    return "act"


__all__ = ["classify_step_phase"]
