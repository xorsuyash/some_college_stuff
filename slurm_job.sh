#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --partition=${SLURM_PARTITION}      
#SBATCH --nodelist=${SLURM_NODELIST}       
#SBATCH --gres=gpu:1                       
#SBATCH --cpus-per-task=16                  
#SBATCH --mem=128G                         
#SBATCH --time=24:00:00                    
#SBATCH --output=logs/%x_%j.out             
#SBATCH --error=logs/%x_%j.err            


module load cuda/12.4 
module load python/3.11  


source ~/envs/llama/bin/activate  


export HF_HOME=~/.cache/huggingface
export WANDB_API_KEY=${WANDB_API_KEY}  


wandb login $WANDB_API_KEY

python trainer.py --num_gpus=1
