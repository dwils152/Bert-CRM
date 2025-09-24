#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=human_bert
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks-per-node=2  # Assuming 4 GPUs per node
#SBATCH --nodes=1            # Number of nodes
#SBATCH --time=100:00:00
#SBATCH --mem=300GB              # Adjust if necessary
#SBATCH --gres=gpu:2         # Request 4 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=str-gpu21


echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

# Code starts here -----------------------------------------

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
#LOCAL_RANK=$SLURM_LOCALID
python -m torch.distributed.run --nnodes=${SLURM_NNODES} --nproc_per_node=2 predict.py --model_path model.pth --data_path chr12.fa --is_distributed
#python -m torch.distributed.run --nnodes=${SLURM_NNODES} --nproc_per_node=4 final.py
#python predict.py --model_path model.pth --data_path chr12_labels.csv --is_distributed


cd $SLURM_SUBMIT_DIR
echo "Hello World! I ran on compute node $(/bin/hostname -s)"
echo ""
echo "======================================================"
echo "End Time   : $(date)"
