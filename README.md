# Intro to Computer Vision ML using YOLOv8 @DSA2024

© DSA2024. Apache License 2.0.

**Introduction**

As a branch of artificial intelligence, computer vision utilizes machine learning and neural networks to guide computers in extracting valuable insights from digital images, videos, and visual data. This capability allows them to offer recommendations or trigger actions when identifying flaws or issues [[1]](https://www.ibm.com/topics/computer-vision).

Computer Vision is a broad concept. It covers more than 15 different applications [[2]](https://huggingface.co/datasets). Here we will look at 3 variations of computer vision, these are : `Image Classification`,`Object Detection` and `Instance Segmentation`. The variations will be `pretrained` and `fine-tuned` prediction. Zero-Shot classification is use of pretrained models to obtain predictions without training.

Here we will use [DSAIL-Porini](https://data.mendeley.com/datasets/6mhrhn7rxc/6) and a bone xray dataset to go through a typical end-to-end machine learning workflow for classification, detection and segementation.


![cls-det-seg](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/2692c40b-591e-4a77-b700-04d51e055b71)

**Topics:**

Content: `Computer Vision`, `YOLOv8`

Level: `Beginner`

**Learning Objectives:**
- Introduce you to end-to-end machine learning.

**Prerequisites:**
- Basic knowledge of [Python Programming](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)
- A [Google Colab](https://colab.research.google.com/) account
- A [roboflow](https://app.roboflow.com/login) account
- A [Hugging Face](https://huggingface.co/join)  account


<!-- #region -->
### A. Image Classification with YOLOv8

Here we shall utilize DSAIL-Porini images to obtain classification predictions and evaluate the model's performance. Luckily, the dataset authors have provided the images and classifications. I'll take you through how to annotate using roboflow.


Here we'll use YOLOv8-cls to obtain predictions for Zebra, Impala and Other classes. We're only going to use images that have a single species of animal and the animals are detectable by YOLOv8-det.

Steps
1. We'll first practice with Image Classification prediction using pretrained `yolov8-cls` model <a target="_blank" href="https://colab.research.google.com/github/Marconi-Lab/dsa_2024/blob/main/dsa2024_yolov8_classification_zero_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

From the test image you can see that the pretrained model has a low confidence score, we'll rectify this by finetuning using annotated data.

2. Download the [zipped dataset](https://github.com/Marconi-Lab/dsa_2024/blob/main/images-cls.zip) and unzip

3. Login to roboflow and create a single-label classification project
![classification_annotation_roboflow](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/0732fdf1-07d1-453d-a9d7-5beb9367321b)

4. Upload the sample images from [DSAIL-Porini](https://data.mendeley.com/datasets/6mhrhn7rxc/6) on roboflow, then save and continue.
![Screenshot 2024-04-28 135625](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/2f1ff7d7-304b-486e-9dd4-11a9b288bd67)

6. Click annotate then annotate images for FineTuning using [roboflow](https://app.roboflow.com/) ... annotate with the classes you see fit. 
![Screenshot 2024-04-28 140838](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/8423ffd0-a378-47c1-907d-c0b6837daea3)

7. Get raw url to annotated [roboflow dataset](https://app.roboflow.com/ds/U8eETZqOAo?key=9AAIElFVFm).
![get-raw-url](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/60868580-6756-4798-91c1-c21ecb9c157f)

8. Train YOLOv8-cls model for Image Classification <a target="_blank" href="https://colab.research.google.com/github/Marconi-Lab/dsa_2024/blob/main/dsa2024-yolov8-classification-training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and you can export model to [**Huggingface**](https://huggingface.co/) (optional). 
9. ReCAP: above we have finetuned `yolov8-cls` for classifying between Zebra, Impala and Other classes.
10. Practice Work
   - Try this process on your annotated dataset.
   - Differentiate zero-shot, one-shot and few-shot training.

<!-- #endregion -->

### B. Image Object Detection with YOLOv8

In the object detection task, we will finetune a YOLOv8 detection model on a camera trap dataset to obtain predictions and characterise the model.

**Steps**

1. Download a pre-annotated subset of the DSAIL-Porini dataset available on [Roboflow](https://universe.roboflow.com/mltowardsobb/dsail-porini-detection-v2).
2. Perform an exploratory data analysis of the dataset using [Fiftyone](https://docs.voxel51.com/) and [Data-Gradients](https://github.com/Deci-AI/data-gradients/)
3. Try out the [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) Zero-shot open vocabulary detection model on the dataset.
4. Fine tune an Ultralytics YOLOv8 model on the dataset
5. Along the way, there are some exercises which you'll do such as trying out the [`Grounding Dino`](https://github.com/IDEA-Research/GroundingDINO) zero-shot model and annotating a few images to finetune a YOLOv8 model with your own version of the dataset.
   
### C. Image Instance Segementation with YOLOv8

For the segmentation task, we will use an open source [xray dataset](https://universe.roboflow.com/bonefrac/seg-2-full/dataset/10) from roboflow.

**Steps**

1. Download the dataset as a zipped file in YOLOv8 format 
2. Unzip the file
3. Log in to Roboflow and create a project ![](https://github.com/Marconi-Lab/dsa_2024/blob/main/assets/Screenshot%20(1).png)
4. Upload your unzipped file ![](https://github.com/Marconi-Lab/dsa_2024/blob/main/assets/Screenshot%20(3).png)
5. Save you dataset but apply pre-processing and augmentations of your choice from the list provided.
6. You can export your dataset again or us the raw url of the dataset. ![](https://github.com/Marconi-Lab/dsa_2024/blob/main/assets/Screenshot%20(5).png)
7. Train YOLOv8-seg model for Segmentation  <a target="_blank" href="https://colab.research.google.com/drive/1xssxI7c9fvIi1K0h_HqeT_U-BsnrMRSL?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
8. In the notebook shared above, we will have finetuned 'yolov8-seg' to carry out instance segmentation for normal vs fractured bones.
9. In the notebook, there are some exercises for you to carry out so as to familiarise yourself more with YOLOv8.


**References**

1. [IBM Computer Vision](https://www.ibm.com/topics/computer-vision)
2. [HuggingFace CV examples](https://huggingface.co/datasets)
3. [Roboflow "What is Zero-Shot Classification"](https://blog.roboflow.com/what-is-zero-shot-classification/#:~:text=Zero%2Dshot%20classification%20models%20are,CLIP)
4. [How to label data for YOLOv5 Instance Segmentation training](https://roboflow.com/how-to-label/yolov5-segmentation)

**Inspiration**

1. [Deep Learning Indaba intro to JAX](https://github.com/deep-learning-indaba/indaba-pracs-2022/blob/main/practicals/Introduction_to_ML_using_JAX.ipynb)
