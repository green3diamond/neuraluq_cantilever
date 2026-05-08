#!/bin/bash
# Submit VI data-only jobs as a SLURM array (max 7 concurrent nodes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FOLDER_NAMES=("results_data_no_ic_v16" "results_data_no_ic_v17")

# Collect data files into an array
# DATA_FILES=($(ls "$SCRIPT_DIR/cantilever/test_scenarios/"*.npy | xargs -n1 basename))
DATA_FILES=(
    "test_NT10_M3_noise001_normal.npy"
    "test_NT11_M3_noise001_uniform.npy"
    "test_NT13_M3_noise003_normal.npy"
    "test_NT14_M3_noise003_uniform.npy"
    "test_NT16_M3_noise005_normal.npy"
    "test_NT17_M3_noise005_uniform.npy"
    "test_NT21_M2_noise005_beta.npy"
    "test_NT22_M3_noise001_beta.npy"
    "test_NT23_M3_noise003_beta.npy"
    "test_NT24_M3_noise005_beta.npy"
)
# DATA_FILES="test_NT01_M2_noise001_normal.npy"
NUM_FILES=${#DATA_FILES[@]}
NUM_FOLDERS=${#OUTPUT_FOLDER_NAMES[@]}
NUM_TASKS=$((NUM_FILES * NUM_FOLDERS))

# Write task list: each line is "data_file output_folder"
TASK_LIST="$SCRIPT_DIR/cantilever/test_scenarios/.task_list.txt"
> "$TASK_LIST"
for OUTPUT_FOLDER_NAME in "${OUTPUT_FOLDER_NAMES[@]}"; do
    for DATA_FILE in "${DATA_FILES[@]}"; do
        echo "${DATA_FILE} ${OUTPUT_FOLDER_NAME}" >> "$TASK_LIST"
    done
done

echo "Found $NUM_FILES data files x $NUM_FOLDERS output folders = $NUM_TASKS tasks"
echo "Submitting array job (max 7 concurrent)..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vi_data
#SBATCH --partition=compute
#SBATCH --array=0-$((NUM_TASKS - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/cantilever/logs/vi_%a_%j.out

eval "\$(\$HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/env

cd /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/cantilever

TASK_LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
DATA_FILE=\$(echo "\$TASK_LINE" | cut -d' ' -f1)
OUTPUT_FOLDER_NAME=\$(echo "\$TASK_LINE" | cut -d' ' -f2)
echo "Task \$SLURM_ARRAY_TASK_ID: \$DATA_FILE -> \$OUTPUT_FOLDER_NAME"

python -u cantilever_vi_data_only.py "test_scenarios/\${DATA_FILE}" "\${OUTPUT_FOLDER_NAME}"
EOF

echo "Submitted array job for $NUM_TASKS tasks"
