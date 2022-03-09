DATASET=$1 # Available datasets: lsun_bedroom, ffhq, lsun_cat, lsun_horse 

wget https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/models/mae_checkpoints/${DATASET}.pth -P checkpoints/mae