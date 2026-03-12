#!/bin/bash
# Submit VI data-only jobs for each data file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILES=($(ls "$SCRIPT_DIR/cantilever/test_scenarios/"*.npy | xargs -n1 basename))
OUTPUT_FOLDER_NAME="results_data_no_ic_v4"

for DATA_FILE in "${DATA_FILES[@]}"; do
    TAG="${DATA_FILE%.npy}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vi_${TAG}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/cantilever/logs/${OUTPUT_FOLDER_NAME}/vi_${TAG}_%j.out

eval "\$(\$HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/env

cd /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/cantilever

python -u cantilever_vi_data_only.py "test_scenarios/${DATA_FILE}" "${OUTPUT_FOLDER_NAME}"
EOF
    echo "Submitted job for ${DATA_FILE}"
done
