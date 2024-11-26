## Prepare the dataset for training, validation and test

set -e

# prepare folders
mkdir -p cache && mkdir -p data && mkdir -p processed_data


## download the dataset
wget -P cache http://avocat.ovh.hw.ipol.im/static/yanhao/masksim/data.tar.gz
tar -xzvf cache/data.tar.gz


# prepare training and validation data
mkdir -p processed_data/train
ln -s $(realpath data/coco_train) processed_data/train/
ln -s $(realpath data/coco_val) processed_data/train/coco_val
ln -s $(realpath data/dresden) processed_data/train/dresden
ln -s $(realpath data/hdrburst) processed_data/train/hdrburst
ln -s $(realpath data/mit5k) processed_data/train/mit5k
ln -s $(realpath data/newsynth) processed_data/train/newsynth


# preprocess test data
python preprocess.py


# Optional: remove the cache/ folder where the zipped files are downloaded
rm -r cache/
