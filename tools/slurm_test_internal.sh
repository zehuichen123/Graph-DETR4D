#!/usr/bin/env bash

set -x

PARTITION=${PARTITION:-AD-RoadUser}
JOB_NAME=${JOB_NAME:-Test}
CONFIG=$1
CHECKPOINT=$2
VIS_RATE=${VIS_RATE:-1}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:3}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test_internal.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n1 \
    python -u internal_code/eval_vis.py ${CONFIG} --sample_rate ${VIS_RATE} ${PY_ARGS}
