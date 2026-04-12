"""Inspect AI plugin entry point.

Inspect discovers extensions via the ``inspect_ai`` entry point group
declared in ``pyproject.toml``. Importing this module triggers registration
of all scorers and metrics shipped with this package.

Side-effecting imports are confined to this module so the rest of the
package never relies on import-time registration.
"""

from __future__ import annotations


def _register() -> None:
    try:
        # Importing the integration package executes the @scorer / @metric
        # decorators, which is what registers them with Inspect AI.
        import inspect_degradation.integration  # noqa: F401
    except ImportError:
        # Inspect AI not installed in this environment; the offline grader
        # and analysis layers remain usable.
        return


_register()
