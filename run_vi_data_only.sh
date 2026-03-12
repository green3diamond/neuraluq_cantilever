#!/bin/bash
# Submit VI data-only jobs for each data file

DATA_FILES=(
    "test_NT02_M2_noise001_uniform.npy"
    "test_NT05_M2_noise003_uniform.npy"
    "test_NT08_M2_noise005_uniform.npy"
)

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
#SBATCH --output=/mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/examples/results_data_no_ic/vi_${TAG}_%j.out

eval "\$(\$HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/env

cd /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/examples

python -u cantilever_vi_data_only.py "${DATA_FILE}"
EOF
    echo "Submitted job for ${DATA_FILE}"
done
