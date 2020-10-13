# Tensorflow Certification Study Guide

This is repo is in development. It is used to keep resources, course references, and code examples while preparing for the TensorFlow Developer Certification exam. If the work here helps you in some way please feel free to share, fork, or star.


## Getting Started

### TensorFlow Guide
Strongly recommend reading the following documents before starting.
TensorFlow Certificate Home
https://www.tensorflow.org/certificate

Environment Setup
https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf?authuser=4

Certificate Handbook
https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook.pdf

### Blog Articles
The following blog articles helped get an idea what others did in terms of study for the exam.

This was the first article I found that inspired me to take on the exam. Thanks Harshit!
https://www.freecodecamp.org/news/how-i-passed-the-certified-tensorflow-developer-exam/

Have followed Daniel Bourke's content since I started into Machine Learning.
https://towardsdatascience.com/how-i-passed-the-tensorflow-developer-certification-exam-f5672a1eb641

Roberto's perspective on the exam is personal, fresh, and insightful.
https://medium.com/@rbarbero/tensorflow-certification-tips-d1e0385668c8

## Courses
Note on courses, you want to focus on implementing and wrtiting as many models as possible with TensorFlow. The couses below were the ones I found to have the most hands on content.

1. [Udacity Intro To Tensorflow](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
    Good if you like structure and are starting with no knowledge. 
2. [Coursera Tensorflow Developer Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)
    Everyone should take this course. This was my primary reference for study.
3. [DataCamp Introduction to Deep Learning](https://www.datacamp.com/courses/introduction-to-deep-learning-with-keras)
    Good for getting started and building muscle memory with the Keras API.
4. [DataCamp Image Processing With Keras](https://www.datacamp.com/courses/image-processing-with-keras-in-python)
    Good overview of simple image models.
5. [DataCamp Advanced Deep Learning With Keras](https://www.datacamp.com/courses/advanced-deep-learning-with-keras)
    Shows you how to use uncommon architectures in Keras. Found this highly useful for rounding out my knowledge of the Keras API.
    
    
## Skills Checklist & TensorFlow Examples
Collection of TensorFlow Models I built while studying for the exam. For each model I mapped the skills requirements from the TensorFlow study handbook.
1. Tensorflow Basics. Code. Kaggle Notebook.
Regression
https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/data
Classification
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

    - Building, compiling, training, evaluating models for binary and muilti- classification
    - Identifying and mitigating overfitting
    - Plotting loss and accuracy
    - Matching input and output shapes
    - Using early stopping callbacks
    - Using datasets from tensorflow datasets
2. Image Classification. Code. Kaggle Notebook.

https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview
- 1 vs many (binary problem)
- Full multi-classification

    - Using Conv2D and MaxPooling Layers
    - Understanding how convolutions improve image the nerual network
    - Using image augmentation and improving overfitting
    - Use ImageDataGenerator and the directory labelling structure
    - Using transfer learning and model checkpoints
3. Natural Language Processing. Code. Kaggle Notebook.

https://www.kaggle.com/c/tweet-sentiment-extraction
https://www.kaggle.com/c/quora-insincere-questions-classification
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
https://www.kaggle.com/c/whats-cooking-kernels-only/overview/evaluation

    - Prepare text for use in a TensorFlow model
    - Use TensorFlow to identify text in binary and multi-class categorization.
    - Use RNNs, LSTMs, and GRUs
    - Train word embeddings and import word embedding weights
    - Train LSTMs on existing text and generate text
4. Comparing TensorFlow Models in a Time Series Forecasting Task. Code. [Kaggle Notebook](https://www.kaggle.com/nicholasjhana/multi-variate-time-series-forecasting-tensorflow).
    - Using RNNs and CNNs in forecasting.
    - Identify when to use trailing and centred windows
    - Adjusting the learning rate with a Learning Rate Scheduler
    - Preparing features and labels
    - Identify and compensate Sequence Bias


## TensorFlow Inputs and Outputs Table
In development.