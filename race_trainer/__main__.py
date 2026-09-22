import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .cli import main  # noqa: E402

raise SystemExit(main())
