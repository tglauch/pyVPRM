#!/bin/bash
#SBATCH --account=mj0143
#SBATCH --job-name=modis_dl
#SBATCH --partition=compute
#SBATCH --output=logs/modis_%A_%a.out
#SBATCH --error=logs/modis_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

source /sw/spack-levante/jupyterhub/jupyterhub/etc/profile.d/conda.sh
conda activate /work/mj0143/b301108/conda/envs/pyvprm_env

cd /work/mj0143/b301108/pyVPRM/pyVPRM/sat_managers
export PYTHONPATH=/work/mj0143/b301108/pyVPRM/pyVPRM:$PYTHONPATH

OUTPUT_DIR=/work/mj0143/b301108/pyVPRM/pyVPRM/sat_managers/tiles_MCD43A4_all/
TASKFILE=tasklist_tile_year.json
IDX=$((SLURM_ARRAY_TASK_ID - 1))

TILE=$(jq -r ".[$IDX].tile" "$TASKFILE")
YEAR=$(jq -r ".[$IDX].year" "$TASKFILE")
PRODUCT=$(jq -r ".[$IDX].product" "$TASKFILE")

TOKEN=$(cat token.txt)

python download_modis_new_parallel.py \
    --tile "${TILE}" --year "${YEAR}" --token "${TOKEN}" \
    --output "${OUTPUT_DIR}" --product "${PRODUCT}" \
    --workers ${SLURM_CPUS_PER_TASK}
