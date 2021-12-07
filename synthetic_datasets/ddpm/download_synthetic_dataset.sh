DATASET=$1

wget -c https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/synthetic-datasets/ddpm/${DATASET}.tar.gz
tar -xzf ${DATASET}.tar.gz -C synthetic_datasets/ddpm/
rm ${DATASET}.tar.gz