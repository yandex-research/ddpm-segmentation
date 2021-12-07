# Notes:
# * This setting is used to load a single machine with 8xA100 80Gb. Please adjust the arguments for your infrastracture.
# * The synthetic dataset is saved to the experiment folder, where the ensemble models are placed. (Produced by train_interpreter.sh) 

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 100 --num_samples 10000 --timestep_respacing 1000 --use_ddim False"

NUM_GPUS="8"
DATASET=$1 # Available datasets: bedroom_28, ffhq_34, cat_15, horse_21

echo "Generating a synthetic dataset..."
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS generate_dataset.py --exp experiments/${DATASET}/datasetDDPM.json $MODEL_FLAGS $SAMPLE_FLAGS 

 