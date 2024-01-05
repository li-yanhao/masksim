# MaskSim: Detection of synthetic images by masked spectrum similarity analysis

The code is being completed on the fly :rocket:



The pretrained model weights can be downloaded [here](https://cirrus.universite-paris-saclay.fr/s/SscHmgDi2gyiF2s). Then unzip the weight files in the `checkpoints` folder like below:
```
checkpoints
|--compress_Q70
|--compress_Q80
|--compress_Q90
|--uncompress
```



To test the program on one single image:
``` sh
python detect_one_image.py -i fake_img.png
```



