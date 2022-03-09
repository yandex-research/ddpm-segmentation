export OMP_NUM_THREADS=1

DATASET=$1 # Available datasets: bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30

python train_interpreter.py --exp experiments/${DATASET}/mae.json