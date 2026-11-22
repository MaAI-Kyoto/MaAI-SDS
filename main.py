"""`main.py` の互換シム。実体は `src/maai_sds/main.py`。

`uv run python main.py` という既存の起動方法を壊さないために残している。
パッケージとしては `python -m maai_sds.main` / `maai-sds` でも起動できる。
"""

import sys
from pathlib import Path

# editable install されていない環境でも動くよう src/ を通す
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maai_sds.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
