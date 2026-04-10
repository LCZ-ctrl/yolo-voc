# YOLOv3-VOC

This is a modified **YOLOv3** implementation.

| Model  | Train Dataset                       | Val Dataset  | Epochs | Input Size  | Test Size | mAP@0.5 | mAP@0.6 | mAP@0.75 |
|:-------|:------------------------------------|:-------------|:-------|:------------|:----------|:--------|:--------|:---------|
| YOLOv3 | VOC2007 trainval + VOC2012 trainval | VOC2007 test | 80     | multi-scale | 416x416   | 81.57%  | 75.89%  | 54.48%   |

## Structure

```
├── data/
|   └── VOCdevkit
├── model/
|   ├── __init__.py
|   ├── yolov3_backbone.py
|   ├── yolov3_neck.py
|   ├── yolov3_fpn.py
|   ├── yolov3_head.py
|   └── yolov3.py
├── config.py
├── voc.py
├── augmentation.py
├── matcher.py
├── loss.py
├── eval.py
├── train.py
└── test.py
```

<em>Read files in order:</em>
> config.py -> yolov3.py -> voc.py -> augmentation.py -> matcher.py -> loss.py -> eval.py -> train.py -> test.py

## Some Results

<br>
<p align="center">
  <img src="./images/009734.jpg" height="180" />
  <img src="./images/009763.jpg" height="180" />
  <img src="./images/000725.jpg" height="180" />
  <br>
  <img src="./images/005676.jpg" height="180" />
  <img src="./images/005698.jpg" height="180" />
  <img src="./images/006621.jpg" height="180" />
</p>

## What's new?

#### <em>Backbone Network</em>:

*Darknet-53* plays a critical role in the performance of YOLOv3 object detection system. It comprises 53 convolutional
layers, making it deeper and more powerful. This increase in depth allows the
network to capture more complex features, improving its detection capabilities.
<br>
<p align="center">
  <img src="./images/darknet53.jpg" height="400" />
  <br>
  <em><strong>DarkNet-53</strong></em>
</p>

#### <em>Multi-level Detection & FPN</em>:

For a Convolutional Neural Network (CNN), as the layers get deeper and the downsampling increases, feature maps at
different depths naturally carry different levels of spatial information (localization) and semantic information (
classification).

Feature maps from shallower layers haven't been "over-processed" by many convolutions, so their semantic information is
relatively low. However, because they haven't gone through much downsampling, they
retain rich spatial information. In contrast, deeper feature maps are the exact opposite. After passing through plenty
of layers, they have richer semantic informations, but the spatial
info gets weaker by too much downsampling. This leading to poor performance on small object detection. At the same time,
as the depth increases, the
receptive field grows, allowing the network to learn large objects more fully, which generally improves large object
detection ability.

After recognizing this trade-off, a simple solution is: let shallow features handle small objects, and let deep features
take care of those large ones. Feature Pyramid Networks (FPN) introduces a top-down feature fusion structure, using
spatial upsampling to continuously integrate high-level semantic information from deep layers into shallower feature
maps.
<br>
<p align="center">
  <img src="./images/fpn.jpg" height="250" />
  <br>
  <em><strong>Feature Pyramid Networks (FPN)</strong></em>
</p>

Here, I used three feature maps C3, C4, and C5 with downsampling strides of 8, 16, and 32. For each feature map, three
anchor boxes are assigned to every grid cell:

- For C3 feature map, anchors (10,13), (16,30), and (33,23) are used for detecting small objects.
- For C4 feature map, anchors (30,61), (62,45), and (59,119) are used for detecting medium-sized objects.
- For C5 feature map, anchors (116,90), (156,198), and (373,326) are used for detecting large objects.

## Train

To start training, run the command -

```
python train.py
```

I used Automatic Mixed Precision (AMP) to accelerate the training process and reduce memory consumption without
sacrificing numerical precision. Furthermore, I used a Cosine Annealing scheduler with a linear warm-up phase during
training. Additionally, Multi-scale Training was implemented, where the input image resolution was randomly sampled
every epoch.

<br>
<p align="center">
  <img src="./images/yolov3_training_metrics.png" height="300" />
  <br>
  <em><strong>Loss and mAP@0.5</strong></em>
</p>

## Test

To test your trained model, run the command -

```
python test.py
```

It will randomly select an image in the test set, and then output the model's prediction results. You can also try your
own images!

<br><br>
<em><strong>My pre-trained
model:</strong></em> [YOLOv3](https://drive.google.com/file/d/17F122qQvdsfd3r2e0SWNeMX4f5rEJLQt/view?usp=drive_link)
