# YogAI

YogAI is a virtual yoga instructor on a raspberry pi smart mirror. Using an Openpose tensorflow implementation forked from [ildoonet/tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), we can guide and instruct a student during their yoga session and improve their form.

*** We've have a tflite implementation! For much faster inference, please see this [repo](https://github.com/mayorquinmachines/YogAI) ***

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk

#### Hardware:

- raspberry pi 3
- any webcam of choice
- a speaker with aux cord
- computer screen
- one way mirror + frame (optional)

### Install

```bash
$ git clone https://www.github.com/ildoonet/YogAI
$ cd YogAI
$ pip3 install -r requirements.txt
```

## Models

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

### Gathering pose data

The [Hackster](https://www.hackster.io/smellslikeml/yogai-smart-personal-trainer-f53744) post will show you how to obtain training samples for your desired poses. Use the ``` yoga_pose_data.py``` script to transform the images into Posenet point arrays with labels. The ``` YogAI_knn.ipynb``` is a jupyter notebook to help you train a KNN to classify yoga poses. 


## Training

See : [etcs/training.md](./etcs/training.md)

## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

[4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

### Lifting from the deep

[1] Arxiv Paper : https://arxiv.org/abs/1701.00295

[2] https://github.com/DenisTome/Lifting-from-the-Deep-release

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2
