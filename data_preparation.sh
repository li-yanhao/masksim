set -e

# prepare folders
mkdir -p cache && mkdir -p data && mkdir -p processed_data

# download pristine images
wget -P cache https://cirrus.universite-paris-saclay.fr/s/2eabgG8fZy8nXME/download/train.zip
unzip cache/train.zip -d cache/
mv cache/train/* data/
rm -r cache/train

# download synthbuster dataset
wget -P cache https://zenodo.org/records/10066460/files/synthbuster.zip
unzip cache/synthbuster.zip -d data/

# download newsynth dataset
python download.py

# prepare training and validation data
mkdir -p processed_data/train
ln -s $(realpath data/coco_train) processed_data/train/
ln -s $(realpath data/coco_val) processed_data/train/coco_val
ln -s $(realpath data/dresden) processed_data/train/dresden
ln -s $(realpath data/hdrburst) processed_data/train/hdrburst
ln -s $(realpath data/mit5k) processed_data/train/mit5k
ln -s $(realpath data/newsynth) processed_data/train/newsynth

# prepare evaluation data
python preprocess.py

# Optional: remove the cache/ folder where the zipped files are downloaded
rm -r cache/
