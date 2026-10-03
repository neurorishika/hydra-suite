"""``hydra`` console entry point.

Dispatches headless subcommands BEFORE any Qt import, so ``hydra doctor``
works over ssh, in CI, and on machines with no display. Anything else starts
the GUI launcher exactly as before.
"""

from __future__ import annotations

import sys
from typing import List, Optional

USAGE = """usage: hydra [doctor [--json] [--tier cpu|mps|cuda] ...]

  hydra            open the HYDRA Suite launcher (GUI)
  hydra doctor     verify this installation (headless)
  hydra --version  print the installed version
"""


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "doctor":
        from hydra_suite.runtime.doctor import main as doctor_main

        return doctor_main(args[1:])
    if args and args[0] in ("--version", "-V"):
        from importlib.metadata import version

        print(f"hydra-suite {version('hydra-suite')}")
        return 0
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    from hydra_suite.launcher.app import main as gui_main

    result = gui_main()
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
