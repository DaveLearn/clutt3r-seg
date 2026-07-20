#!/usr/bin/env bash
set -euo pipefail

# Clones the upstream DuoduoCLIP checkout that clutt3rseg/clip_backends/duoduo.py
# imports `src.model.wrapper` from (it is put on sys.path, not pip-installed).
# DUODUOCLIP_ROOT points here; see [tool.pixi.activation].
#
# Nothing is installed here: DuoduoCLIP's requirements.txt (h5py, wandb, pillow,
# lightning, hydra-core, torchmetrics) is already covered by this project's pixi
# dependencies, and its modified open_clip is a pypi dependency (open_clip_torch).
# Keep DUODUOCLIP_REV in sync with the rev pinned there.

INSTALL_DIR="${1:-external/DuoduoCLIP}"
REPO_URL="${DUODUOCLIP_REPO:-https://github.com/3dlg-hcvc/DuoduoCLIP.git}"
REV="${DUODUOCLIP_REV:-3111a77041fc82e656fe589bd7737b41ea34c10d}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    git clone --filter=blob:none "${REPO_URL}" "${INSTALL_DIR}"
fi

git -C "${INSTALL_DIR}" fetch origin "${REV}"
git -C "${INSTALL_DIR}" checkout --detach "${REV}"

echo "DuoduoCLIP ready at $(cd "${INSTALL_DIR}" && pwd)"
