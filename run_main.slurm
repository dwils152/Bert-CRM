#!/bin/bash
#SBATCH --partition=Orion
#SBATCH --job-name=Main-Cross-Species
#SBATCH --output=nf-main.out
#SBATCH --error=nf-main.err
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem=10GB

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

# Code starts here -----------------------------------------
module load samtools
module load bedtools2
module load gcc/12.3.0

# cd BertCRM
# pip install .
# cd ../

export NXF_TEMP=/scratch/dwils152/tmp
export TMPDIR=/scratch/dwils152/tmp

#nextflow run main.nf -resume
#git add BertCRM/* -v


cd $SLURM_SUBMIT_DIR
echo "Hello World! I ran on compute node $(/bin/hostname -s)"
echo ""
echo "======================================================"
echo "End Time   : $(date)"
