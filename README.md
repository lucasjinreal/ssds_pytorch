# SSDs

This repo contains many object detection methods that aims at **single shot and real time**, so the **speed** is the only thing we talk about. Currently we have some base networks that support object detection task such as MobileNet V2, ResNet, VGG etc. And some SSD variants such as FSSD, RFBNet, Retina, and even Yolo are contained.

If you have any faster object detection methods welcome to discuss with me to merge it into our master branches.




# Note

Work are just being progressing. Will update some result and pretrained model after trained on some datasets. And of course, some out-of-box inference demo.

[updates]:

2018.11.06: As you know, after trained `fssd_mobilenetv2` the inference codes actually get none result, still debugging how this logic error comes out.



# Train

All settings about base net and ssd variants are under `./experiments/cfgs/*.yml`, just edit it to your enviroment and kick it off.

```
python3 train.py --cfg=./experiments/cfgs/rfb_lite_mobilenetv2_train_vocyml
```

You can try train on coco first then using your custom dataset. If you have your coco data inside /path/to/coco, the just link it to `./data/` and you can find coco inside `./data`.

![](https://s1.ax1x.com/2018/11/04/i5nrdO.png)

That is what it trains like. After that I shall upload some trained model.
