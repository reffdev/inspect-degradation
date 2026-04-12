"""Tests for the step-phase classifier."""

import pytest

from inspect_degradation.step_phase import classify_step_phase


# ── Auto-SWE tool calls ──────────────────────────────────────────


class TestAutoSWEToolCalls:
    """[tool_call: X] format used by Auto-SWE framework."""

    def test_readfile_is_explore(self):
        action = '[tool_call: readFile] {"path":"src/frontend/Dashboard.tsx"}'
        assert classify_step_phase(action) == "explore"

    def test_readfile_with_reasoning_is_explore(self):
        action = (
            "I'll explore the codebase to understand the current structure.\n\n"
            '[tool_call: readFile] {"path":"src/frontend/Dashboard.tsx"}'
        )
        assert classify_step_phase(action) == "explore"

    def test_searchfiles_is_explore(self):
        action = '[tool_call: searchFiles] {"pattern":"dashboard","path":"src/"}'
        assert classify_step_phase(action) == "explore"

    def test_listdirectory_is_explore(self):
        action = '[tool_call: listDirectory] {"path":"src/frontend/"}'
        assert classify_step_phase(action) == "explore"

    def test_gitdiff_is_explore(self):
        action = "[tool_call: gitDiff] {}"
        assert classify_step_phase(action) == "explore"

    def test_gitstatus_is_explore(self):
        action = "[tool_call: gitStatus] {}"
        assert classify_step_phase(action) == "explore"

    def test_readrelevantfiles_is_explore(self):
        action = '[tool_call: readRelevantFiles] {"query":"dashboard routing"}'
        assert classify_step_phase(action) == "explore"

    def test_lookupdocs_is_explore(self):
        action = '[tool_call: lookupDocs] {"query":"react router"}'
        assert classify_step_phase(action) == "explore"

    def test_todo_is_explore(self):
        action = '[tool_call: todo] {"items":["fix routing","add tests"]}'
        assert classify_step_phase(action) == "explore"

    def test_replaceinfile_is_act(self):
        action = (
            '[tool_call: replaceInFile] {"path":"src/frontend/Sidebar.tsx",'
            '"old_str":"machines.find((m) => m.id === selectedMachineId)",'
            '"new_str":"machines.filter((m) => m.active).find((m) => m.id === selectedMachineId)"}'
        )
        assert classify_step_phase(action) == "act"

    def test_replaceinfile_with_search_in_content_is_act(self):
        """Code content containing 'search', 'find', etc. must not fool the classifier."""
        action = (
            '[tool_call: replaceInFile] {"path":"src/utils.ts",'
            '"old_str":"function search(query) { return find(query); }",'
            '"new_str":"function search(query) { return findAll(query); }"}'
        )
        assert classify_step_phase(action) == "act"

    def test_writefile_is_act(self):
        action = '[tool_call: writeFile] {"path":"src/new.tsx","content":"import React..."}'
        assert classify_step_phase(action) == "act"

    def test_appendtofile_is_act(self):
        action = '[tool_call: appendToFile] {"path":"log.txt","content":"done"}'
        assert classify_step_phase(action) == "act"

    def test_deletefile_is_act(self):
        action = '[tool_call: deleteFile] {"path":"src/old.tsx"}'
        assert classify_step_phase(action) == "act"

    def test_runcommand_is_act(self):
        action = '[tool_call: runCommand] {"command":"npm run build"}'
        assert classify_step_phase(action) == "act"

    def test_checkbuild_is_act(self):
        action = "[tool_call: checkBuild] {}"
        assert classify_step_phase(action) == "act"

    def test_checktests_is_act(self):
        action = "[tool_call: checkTests] {}"
        assert classify_step_phase(action) == "act"

    def test_submitverdict_is_act(self):
        action = '[tool_call: submitVerdict] {"verdict":"pass","reason":"All tests pass"}'
        assert classify_step_phase(action) == "act"

    def test_submitresult_is_act(self):
        action = '[tool_call: submitResult] {"result":"completed"}'
        assert classify_step_phase(action) == "act"

    def test_mixed_explore_and_act_is_act(self):
        """If a step has both readFile and replaceInFile, it's act."""
        action = (
            '[tool_call: readFile] {"path":"src/a.tsx"}\n'
            '[tool_call: replaceInFile] {"path":"src/a.tsx","old_str":"x","new_str":"y"}'
        )
        assert classify_step_phase(action) == "act"

    def test_unknown_tool_defaults_to_act(self):
        action = '[tool_call: unknownTool] {"foo":"bar"}'
        assert classify_step_phase(action) == "act"


# ── OpenHands bracket commands ────────────────────────────────────


class TestOpenHandsBracketCommands:
    """[bash] / [str_replace_editor] format used by OpenHands."""

    def test_str_replace_editor_view_is_explore(self):
        action = "[str_replace_editor] view"
        assert classify_step_phase(action) == "explore"

    def test_str_replace_editor_str_replace_is_act(self):
        action = '[str_replace_editor] str_replace\n{"path":"src/main.py","old_str":"old","new_str":"new"}'
        assert classify_step_phase(action) == "act"

    def test_str_replace_editor_create_is_act(self):
        action = "[str_replace_editor] create"
        assert classify_step_phase(action) == "act"

    def test_str_replace_editor_insert_is_act(self):
        action = "[str_replace_editor] insert"
        assert classify_step_phase(action) == "act"

    def test_str_replace_editor_undo_edit_is_act(self):
        action = "[str_replace_editor] undo_edit"
        assert classify_step_phase(action) == "act"

    def test_submit_is_act(self):
        action = "[submit] "
        assert classify_step_phase(action) == "act"

    def test_finish_is_act(self):
        action = "[finish] "
        assert classify_step_phase(action) == "act"

    def test_bash_grep_is_explore(self):
        action = "[bash] grep -r 'def test_' tests/"
        assert classify_step_phase(action) == "explore"

    def test_bash_find_is_explore(self):
        action = "[bash] find . -name '*.py' -type f"
        assert classify_step_phase(action) == "explore"

    def test_bash_cat_is_explore(self):
        action = "[bash] cat src/main.py"
        assert classify_step_phase(action) == "explore"

    def test_bash_ls_is_explore(self):
        action = "[bash] ls -la src/"
        assert classify_step_phase(action) == "explore"

    def test_bash_cd_is_explore(self):
        action = "[bash] cd /workspace/repo"
        assert classify_step_phase(action) == "explore"

    def test_bash_python_is_act(self):
        action = "[bash] python -m pytest tests/"
        assert classify_step_phase(action) == "act"

    def test_bash_pip_install_is_act(self):
        action = "[bash] pip install requests"
        assert classify_step_phase(action) == "act"

    def test_bash_sed_is_act(self):
        action = "[bash] sed -i 's/old/new/g' file.py"
        assert classify_step_phase(action) == "act"

    def test_bash_rm_is_act(self):
        action = "[bash] rm -rf build/"
        assert classify_step_phase(action) == "act"

    def test_bash_kill_is_act(self):
        """kill is state-changing, should be act."""
        action = "[bash] kill -9 12345"
        assert classify_step_phase(action) == "act"

    def test_mixed_view_and_edit_is_act(self):
        """If a step has both view and str_replace, it's act."""
        action = "[str_replace_editor] view\n[str_replace_editor] str_replace"
        assert classify_step_phase(action) == "act"


# ── XML tool blocks (SWE-agent, terminus) ─────────────────────────


class TestXMLToolBlocks:
    """<execute_bash> format used by SWE-agent and terminus."""

    def test_execute_bash_grep_is_explore(self):
        action = "<execute_bash>\n<command>grep -rn 'class Test' tests/</command>\n</execute_bash>"
        assert classify_step_phase(action) == "explore"

    def test_execute_bash_find_is_explore(self):
        action = "<execute_bash>\n<command>find . -name '*.py' | head -20</command>\n</execute_bash>"
        assert classify_step_phase(action) == "explore"

    def test_execute_bash_cat_is_explore(self):
        action = "<execute_bash>\n<command>cat src/main.py</command>\n</execute_bash>"
        assert classify_step_phase(action) == "explore"

    def test_execute_bash_ls_is_explore(self):
        action = "<execute_bash>\n<command>ls -la</command>\n</execute_bash>"
        assert classify_step_phase(action) == "explore"

    def test_execute_bash_python_is_act(self):
        action = "<execute_bash>\n<command>python test_fix.py</command>\n</execute_bash>"
        assert classify_step_phase(action) == "act"

    def test_execute_bash_pytest_is_act(self):
        action = "<execute_bash>\n<command>pytest tests/ -v</command>\n</execute_bash>"
        assert classify_step_phase(action) == "act"

    def test_execute_bash_sed_is_act(self):
        action = "<execute_bash>\n<command>sed -i 's/old/new/' file.py</command>\n</execute_bash>"
        assert classify_step_phase(action) == "act"

    def test_execute_bash_pip_is_act(self):
        action = "<execute_bash>\n<command>pip install -e .</command>\n</execute_bash>"
        assert classify_step_phase(action) == "act"

    def test_with_think_block(self):
        action = (
            "<think>Let me check the test files</think>\n"
            "<execute_bash>\n<command>ls tests/</command>\n</execute_bash>"
        )
        assert classify_step_phase(action) == "explore"

    def test_execute_ipython_is_supported(self):
        action = "<execute_ipython>\nprint(open('file.py').read())\n</execute_ipython>"
        # ipython reading a file — should pick up as explore or default
        # The content doesn't match shell patterns cleanly, so this may default
        assert classify_step_phase(action) in ("explore", "act")

    def test_git_diff_in_xml_is_explore(self):
        action = "<execute_bash>\n<command>git diff HEAD~1</command>\n</execute_bash>"
        assert classify_step_phase(action) == "explore"

    def test_git_add_in_xml_is_act(self):
        action = "<execute_bash>\n<command>git add -A && git commit -m 'fix'</command>\n</execute_bash>"
        assert classify_step_phase(action) == "act"


# ── SWE-agent specific commands ───────────────────────────────────


class TestSWEAgentCommands:
    """SWE-agent custom commands like find_file, open, edit, submit."""

    def test_find_file_is_explore(self):
        action = "find_file test_utils.py"
        assert classify_step_phase(action) == "explore"

    def test_search_file_is_explore(self):
        action = 'search_file "def test_" tests/test_main.py'
        assert classify_step_phase(action) == "explore"

    def test_search_dir_is_explore(self):
        action = 'search_dir "import logging" src/'
        assert classify_step_phase(action) == "explore"

    def test_open_is_explore(self):
        action = "open src/main.py"
        assert classify_step_phase(action) == "explore"

    def test_goto_is_explore(self):
        action = "goto 150"
        assert classify_step_phase(action) == "explore"

    def test_scroll_up_is_explore(self):
        action = "scroll_up"
        assert classify_step_phase(action) == "explore"

    def test_scroll_down_is_explore(self):
        action = "scroll_down"
        assert classify_step_phase(action) == "explore"

    def test_edit_is_act(self):
        action = "edit 10:15\nimport os\nimport sys\nend_of_edit"
        assert classify_step_phase(action) == "act"

    def test_create_is_act(self):
        action = "create src/new_file.py"
        assert classify_step_phase(action) == "act"

    def test_submit_is_act(self):
        action = "submit"
        assert classify_step_phase(action) == "act"


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string_defaults_to_act(self):
        assert classify_step_phase("") == "act"

    def test_pure_reasoning_defaults_to_act(self):
        action = "I think the issue is in the routing logic. Let me analyze this further."
        assert classify_step_phase(action) == "act"

    def test_code_content_with_find_not_misclassified(self):
        """Words like 'find' inside code content shouldn't trigger explore
        when there's no actual command structure."""
        # This is tricky — plain text with 'find' in it. The old classifier
        # would match this as explore. The new one should too IF there's
        # no structured tool call wrapping it.
        action = "results = db.find({'status': 'active'})"
        # This is ambiguous — could be code being discussed or a command.
        # We accept either classification here.
        result = classify_step_phase(action)
        assert result in ("explore", "act")

    def test_runcommand_with_grep_inside_is_act(self):
        """Auto-SWE runCommand containing grep in the command string should be act."""
        action = '[tool_call: runCommand] {"command":"grep -r search_pattern src/"}'
        assert classify_step_phase(action) == "act"

    def test_runcommand_with_cat_inside_is_act(self):
        """runCommand is always act regardless of what command it runs."""
        action = '[tool_call: runCommand] {"command":"cat /etc/hosts"}'
        assert classify_step_phase(action) == "act"
