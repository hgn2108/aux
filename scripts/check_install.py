"""Run the test suite as a plain install would, with the optional extras hidden.

A dependency declared under an optional extra but imported at module level by the library
makes the package uninstallable-in-practice: it imports fine on a development machine,
where the extra happens to be present, and fails everywhere else. That is exactly how the
first CI run broke -- `aux.data` imports pandas to read the FMA metadata CSV, and pandas
was declared under `[app]`.

This hides the optional packages with an import hook and runs the suite, so the failure
shows up locally in a second rather than in CI in two minutes.

    python scripts/check_install.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

#: The optional extras. Anything the library itself needs must not be in here.
OPTIONAL = ("streamlit", "altair")

BLOCKER = textwrap.dedent(f"""
    import sys

    BLOCKED = {set(OPTIONAL)!r}

    class _Hide:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}")
            return None

    sys.meta_path.insert(0, _Hide())
""")


def main() -> int:
    args = sys.argv[1:] or ["-m", "not slow", "-q"]
    code = BLOCKER + f"\nimport pytest; raise SystemExit(pytest.main({args!r}))"
    print(f"running the suite without {', '.join(OPTIONAL)}", file=sys.stderr)
    return subprocess.run([sys.executable, "-c", code]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
