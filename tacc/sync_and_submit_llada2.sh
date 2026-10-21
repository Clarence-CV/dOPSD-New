#!/bin/bash
# Run LOCALLY (WSL) after `ssh -fN ls6`. Syncs this working tree to $WORK/inf385t/dOPSD-sd on Lonestar6, shows the
# queue limits, and submits tacc/llada2_pipeline.slurm from there. Knobs are passed through to the job.
#   SMOKE=1 STAGES="baseline states profile readout joint jointeval" SBATCH_ARGS="-p gpu-a100-small -t 00:45:00" \
#       bash tacc/sync_and_submit_llada2.sh
#   STAGES=baseline SBATCH_ARGS="-t 03:00:00" bash tacc/sync_and_submit_llada2.sh
#   DRY=1 bash tacc/sync_and_submit_llada2.sh        # sync + qlimits only
set -euo pipefail
cd "$(dirname "$0")/.."
ssh -O check ls6 >/dev/null 2>&1 || { echo "no ls6 master connection; run: ssh -fN ls6"; exit 1; }
REMOTE_WORK=$(ssh ls6 'echo $WORK')
DEST="$REMOTE_WORK/inf385t/dOPSD-sd"
rsync -az --delete --exclude '.git/' --exclude 'outputs/' --exclude 'logs/' --exclude '__pycache__/' \
      --exclude 'checkpoints/' --exclude 'wandb/' ./ "ls6:$DEST/"
git rev-parse HEAD > /tmp/sd_rev && git status --short > /tmp/sd_status
scp -q /tmp/sd_rev "ls6:$DEST/.code_revision"; scp -q /tmp/sd_status "ls6:$DEST/.code_status"
echo "synced to $DEST at $(cat /tmp/sd_rev)$( [[ -s /tmp/sd_status ]] && echo ' (+uncommitted changes)')"
ssh ls6 "qlimits 2>/dev/null | grep -E 'Queue|gpu-a100' || true"
PASS=""
for v in RUN STAGES SMOKE EVAL_SPLIT EVAL_SUBSPLIT N_EVAL READOUT_STEPS JOINT_STEPS READOUT_SELECT CONFIG; do
    [[ -n "${!v:-}" ]] && PASS+="$v='${!v}' "
done
if [[ "${DRY:-0}" == "1" ]]; then echo "DRY=1: not submitting"; exit 0; fi
ssh ls6 "cd $DEST && mkdir -p logs && env $PASS sbatch --parsable ${SBATCH_ARGS:-} tacc/llada2_pipeline.slurm" | tee /tmp/llada2_jobid
echo "submitted job $(cat /tmp/llada2_jobid); monitor: ssh ls6 'squeue -u \$USER'"
