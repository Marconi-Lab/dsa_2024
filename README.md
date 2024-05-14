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
A. Image Classification with YOLOv8

Here we shall utilize DSAIL-Porini images to obtain classification predictions and evaluate the model's performance. Luckily, the dataset authors have provided the images and classifications. I'll take you through how to annotate using roboflow.


Here we'll use YOLOv8-cls to obtain predictions for Zebra, Impala and Other classes. We're only going to use images that have a single species of animal and the animals are detectable by YOLOv8-det.

Steps
1. Upload 90 images from [DSAIL-Porini](https://data.mendeley.com/datasets/6mhrhn7rxc/6), 30 images per class, [subdataset]().
2. Image Classification using pretrained `yolov8-cls` model <a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/github/Marconi-Lab/dsa_2024/blob/yuri/dsa2024_yolov8_classification_zero_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.
3. Annotate images for FineTuning using [roboflow](https://app.roboflow.com/)
4. Get url to annotated [roboflow dataset](https://app.roboflow.com/ds/U8eETZqOAo?key=9AAIElFVFm).
5. Train YOLOv8-cls model for Image Classification <a target="_blank" href="https://colab.research.google.com/github/Marconi-Lab/dsa_2024/blob/yuri/dsa2024-yolov8-classification-training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and export model to [**Huggingface**](https://huggingface.co/). 
6. ReCAP: above we have finetuned `yolov8-cls` for classifying between Zebra, Impala and Other classes.
7. Practice Work
   - Try this process on a dataset of your choice.
   - Differentiate zero-shot, one-shot and few-shot training.
<!-- #endregion -->

B. Image Object Detection with YOLOv8

C. Image Instance Segementation with YOLOv8
For the segmentation task, we will use an open source [xray dataset](https://universe.roboflow.com/bonefrac/seg-2-full/dataset/10) from roboflow.
Steps
1. Download the dataset as a zipped file in YOLOv8 format 
2. Unzip the file
3. Log in to Roboflow and create a project
4. Upload your unzipped file
5. Save you dataset but apply pre-processing and augmentations of your choice from the list provided.
6. You can export your dataset again or us the raw url of the dataset.
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
