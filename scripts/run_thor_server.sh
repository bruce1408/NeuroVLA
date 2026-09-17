#!/usr/bin/env bash
# 启动 NeuroVLA policy server（Jetson Thor / vla_docker_env）
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
    docker exec -it vla_docker_env bash -lc "$*"
  fi
}

CMD="
set -e
source ${ENV_FILE}
source \${NEUROVLA_VENV}/bin/activate
export PYTHONPATH=\${NEUROVLA_REPO}:\${PYTHONPATH:-}

if [[ ! -f \${NEUROVLA_CKPT_PATH} ]]; then
  echo 'ERROR: checkpoint not found:' \${NEUROVLA_CKPT_PATH}
  echo 'Run: bash NeuroVLA/scripts/setup_thor_deploy.sh'
  exit 1
fi

echo '>>> Starting NeuroVLA server'
echo '    ckpt:' \${NEUROVLA_CKPT_PATH}
echo '    port:' \${NEUROVLA_PORT:-10093}

cd \${NEUROVLA_REPO}
python deployment/model_server/server_policy.py \\
  --ckpt_path \${NEUROVLA_CKPT_PATH} \\
  --port \${NEUROVLA_PORT:-10093} \\
  --use_bf16
"

run_in "${CMD}"
