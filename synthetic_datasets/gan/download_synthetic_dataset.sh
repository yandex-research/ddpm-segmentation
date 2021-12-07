DATASET=$1

wget -c https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/synthetic-datasets/gan/${DATASET}.tar.gz
tar -xzf ${DATASET}.tar.gz -C synthetic_datasets/gan/
rm ${DATASET}.tar.gz