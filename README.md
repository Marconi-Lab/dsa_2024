# Intro to Computer Vision ML using YOLOv8 @DSA2024

© DSA2024. Apache License 2.0.

**Authors**: Yuri Njathi,... ,

**Reviewers** : ...

**Introduction**

As a branch of artificial intelligence, computer vision utilizes machine learning and neural networks to guide computers in extracting valuable insights from digital images, videos, and visual data. This capability allows them to offer recommendations or trigger actions when identifying flaws or issues [[1]](https://www.ibm.com/topics/computer-vision).

Computer Vision is a broad concept. It covers more than 15 different applications [[2]](https://huggingface.co/datasets). Here we will look at 2 variations of `Image Classification`,`Object Detection` and `Instance Segmentation`. The variations will be `zero-shot` and `fine-tuned` prediction. Zero-Shot classification is use of pretrained models to obtain predictions without training.

Here we will use [DSAIL-Porini](https://data.mendeley.com/datasets/6mhrhn7rxc/6) and a health dataset ! [INSERT] to go through a typical end-to-end machine learning workflow for classification, detection and segementation.


![cls-det-seg](https://github.com/Marconi-Lab/dsa_2024/assets/54037190/2692c40b-591e-4a77-b700-04d51e055b71)

**Topics:**

Content: `Computer Vision`, `YOLOv8`

Level: `Beginner`

**Learning Objectives:**
- Introduce you to end-to-end machine learning.

**Prerequisites:**
- Basic knowledge of [Python Programming](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)


<!-- #region -->
A. Image Classification with YOLOv8

Here we shall utilize DSAIL-Porini images to obtain classification predictions and evaluate the model's performance. Luckily, the dataset authors have provided the images and classifications. I'll take you through how to annotate using roboflow.


Here we'll use YOLOv8-cls to obtain zero-shot predictions for Zebra, Impala and Other classes. We're only going to use images that have a single species of animal and the animals are detectable by YOLOv8-det.

Steps
1. Obtain 600 images from DSAIL-Porini, 100 for each class.
2. Conduct and log ZeroShot Image Classification and performance [Open in Colab link]
3. Annotate some images for FineTuning using [roboflow](https://app.roboflow.com/)
4. Export dataset to [HuggingFace](https://huggingface.co/).
5. Train YOLOv8-cls model and export model to **Huggingface**.
6. Conduct and log Finetuned Image Classification and performance [Open in Colab link]
7. ReCap
8. Practice Work
<!-- #endregion -->

B. Image Object Detection with YOLOv8

C. Image Instance Segementation with YOLOv8

<img src="https://github.com/Marconi-Lab/dsa_2024/assets/54037190/5218314b-a836-499c-b5a9-1b38bd9b75a4" alt="dsalogo" width="200"/>
<img src="https://github.com/Marconi-Lab/dsa_2024/assets/54037190/5cfd6bdc-c1e3-437f-ac65-76e76f38a8bb" alt="marconi_mak" width="200"/>
<img src="https://github.com/Marconi-Lab/dsa_2024/assets/54037190/cfdf77dc-22de-40de-a31d-701e106b2a53" alt="dsaillogo_white" width="200"/>


**References**

1. [IBM Computer Vision](https://www.ibm.com/topics/computer-vision)
2. [HuggingFace CV examples](https://huggingface.co/datasets)
3. [Roboflow "What is Zero-Shot Classification"](https://blog.roboflow.com/what-is-zero-shot-classification/#:~:text=Zero%2Dshot%20classification%20models%20are,CLIP)

**Inspiration**

1. [Deep Learning Indaba intro to JAX](https://github.com/deep-learning-indaba/indaba-pracs-2022/blob/main/practicals/Introduction_to_ML_using_JAX.ipynb)
