#!/bin/bash
#SBATCH -J TestJob               
#SBATCH -c 8                    
#SBATCH --mem=128G
#SBATCH -p h100                  
#SBATCH --gres=gpu:1             
#SBATCH --tmp=5G                 
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=<your-email-address>

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
python run_script.py resnet18