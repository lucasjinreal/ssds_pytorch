# SSDs

This repo contains many object detection methods that aims at **single shot and real time**, so the speed is the only thing we talk about. Currently we have some base networks that support object detection task such as MobileNet V2, ResNet, VGG etc. And some SSD variants such as FSSD, RFBNet, Retina, and even Yolo are contained.


# Note

Work are just being progressing. Will update some result and pretrained model after trained on some datasets. And of course, some out-of-box inference demo.

# Train

All settings about base net and ssd variants are under `./experiments/cfgs/*.yml`, just edit it to your enviroment and kick it off.

```
python3 train.py --cfg=./experiments/cfgs/rfb_lite_mobilenetv2_train_vocyml
```
