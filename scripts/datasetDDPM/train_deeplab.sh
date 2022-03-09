# Arguments:

# --data_path: your path to the synthetic data 
# Our released dataset: synthetic_datasets/bedroom_28/ddpm/samples_256x256x3.npz
# Produced by generate_dataset.sh: set nothing or "". By default, it uses the synthetic dataset from the experiment directory. 

# --max_data: number of synthetic images to use for training (default: 50000).
# One can consider increasing it upto 50000 to get extra 2-4% of mIoU.

# --uncertainty_portion: a portion of samples with most uncertain predictions to remove (default: 0.1)
# 0.2-0.25 sometimes can provide slightly better performance.

# Note: One can use this script for evaluation as well. 
# The evaluation is performed right after the training. 
# The script checks whether all checkpoints are available. 
# If yes, the evaluation starts immediately without retraining. 

DATASET=$1 # Available datasets: bedroom_28, ffhq_34, cat_15, horse_21

CUDA_VISIBLE_DEVICES=7 python train_deeplab.py \
        --data_path synthetic_datasets/ddpm/${DATASET}/samples_256x256x3.npz \
        --max_data 50000 \
        --uncertainty_portion 0.1 \
        --exp experiments/${DATASET}/datasetDDPM.json 