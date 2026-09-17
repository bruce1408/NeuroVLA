#!/usr/bin/env bash
# NeuroVLA WebSocket 冒烟测试（需 server 已在运行）
set -euo pipefail

REPO_ROOT="/workspace/NeuroVLA"
ENV_FILE="${REPO_ROOT}/scripts/thor_deploy.env"
IN_CONTAINER=false
for arg in "$@"; do
  [[ "$arg" == "--in-container" ]] && IN_CONTAINER=true
done

run_in() {
  if [[ "${IN_CONTAINER}" == "true" ]]; then
    eval "$*"
  else
    docker exec vla_docker_env bash -lc "$*"
  fi
}

CMD="
set -e
source ${ENV_FILE}
source \${NEUROVLA_VENV}/bin/activate
export PYTHONPATH=\${NEUROVLA_REPO}:\${PYTHONPATH:-}
cd \${NEUROVLA_REPO}/deployment/model_server
python debug_server_policy.py \\
  --host 127.0.0.1 \\
  --port \${NEUROVLA_PORT:-10093} \\
  --test infer
"

run_in "${CMD}"
