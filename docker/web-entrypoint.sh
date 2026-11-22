#!/bin/bash
# web サービスの起動。名前付き volume が /app/.venv を隠す初回はイメージのシードを展開する。
set -euo pipefail
if [ ! -x /app/.venv/bin/python ]; then
    echo "[maai-sds] /app/.venv が空のためイメージのシードを展開します"
    mkdir -p /app/.venv
    cp -a /opt/venv-seed/. /app/.venv/
fi
module="${MAAI_SDS_ENTRYPOINT_MODULE:-maai_sds.web.server}"
exec /app/.venv/bin/python -m "$module" "$@"
