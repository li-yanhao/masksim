# MaskSim: Detection of synthetic images by masked spectrum similarity analysis



## Test

The pretrained model weights can be downloaded [here](https://cirrus.universite-paris-saclay.fr/s/SscHmgDi2gyiF2s). Then unzip the weight files in the `checkpoints` folder like below:
```
checkpoints
|--JPEG_Q70
|--JPEG_Q80
|--JPEG_Q90
|--uncompressed
```


To test the program on a single image:
``` sh
python detect_one_image.py -i fake_img.png
```

To test the program on a JPEG image compressed at quality factor 90:
``` sh
python detect_one_image.py -i fake_img.png -c 90
```


An [IPOL demo](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000482) is now available.


## Training

Download pristine images from [Raise-2k](http://loki.disi.unitn.it/RAISE/download.html), Dresden and the curated subset of [HDR+Burst](https://hdrplusdata.org/dataset.html).

Download synthetic images from [Synthbuster](https://zenodo.org/records/10066460) dataset.

Step 1: Preprocess the images

``` sh
python preprocess.py
```

Step 2: Train the model

``` sh
python train.py  # train the model for uncompressed images
python train.py -Q 90  # train the model for jpeg-compressed images at quality 90
python train.py -Q 80
python train.py -Q 70
```

This will take about 14 hours using an A100 GPU. The weights are saved in `checkpoints`.



Then to evaluate the trained model:
``` sh
python evaluate.py  # no jpeg compression
python evaluate.py -Q 90  # jpeg compression at quality 90
python evaluate.py -Q 80
python evaluate.py -Q 70
```

To be continued :rocket:


