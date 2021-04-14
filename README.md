# Tensorflow Certification Study Guide

This is repo is in development. It is used to keep resources, course references, and code examples while preparing for the TensorFlow Developer Certification exam. If the work here helps you in some way please feel free to share, fork, or star.

**Resource Legend**
- Useful            :bulb:
- Highly Important  :key:
- Essential         :star2:


## Getting Started

### TensorFlow Guide
The following documents are issued by the Tensorflow Developemnt Ceritifcation website. They outline everything tested on the exam, how the exam is conducted, how to register, and how you can prepare.
- [TensorFlow Certificate Home](https://www.tensorflow.org/certificate)
- [Environment Setup:](https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf?authuser=4) Exam takes place in a PyCharm environment. 
- [Certificate Handbook](https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook.pdf) :star2:

### TensorFlow Intro
- [Ultimate Guide To TensorFlow 2.0](https://rubikscode.net/2019/04/22/ultimate-guide-to-tensorflow-2-0-in-python/): A good genric intro to TensorFlow 2.0 with an example on the Iris Dataset.
- Effective TF2](https://www.tensorflow.org/guide/effective_tf2): Google's detailed and indepth guide to all things TensorFlow 2.0. :bulb: 

### Blog Articles
Below is a list of blog articles that help get an idea what others did in terms of study for the exam.

[How I passed the certified tensorflow developer exam, Harshi Tyagi](https://www.freecodecamp.org/news/how-i-passed-the-certified-tensorflow-developer-exam/) 
- Does a good job breaking down his study plan, and reviews each course he completed. Find his resource list [here](https://www.notion.so/15049893501f4387893a5de0059ef8a5?v=9154c52a61494668b12802f157bce0d4)

[How I passed the tensorflow developer certification exam, Daniel Bourke](https://towardsdatascience.com/how-i-passed-the-tensorflow-developer-certification-exam-f5672a1eb641) 
- Any blog article by Daniel Bourke is going to help you get motivated to learn and acheive. Somewhat of a rehash of the previous article, still a quality resource for learning about the exam. He also discusses what happens during the exam and shares his personal exam curriculum in another post [My Machine Learning Curriculum for May 2020: Getting TensorFlow Developer Certified](https://www.mrdbourke.com/ml-study-may-2020/)

[I just passed the TensorFlow certification...here are some tips for you, Roberto Barbero](https://medium.com/@rbarbero/tensorflow-certification-tips-d1e0385668c8) 
- Roberto's perspective on the exam is personal, fresh, and insightful. Recommended for his commentary on how to work through problems as they arrise on the exam.

## Discussions
1. [Outline of exam topics detailed](https://www.kaggle.com/questions-and-answers/183715)
2. [Summary and description of how one Kaggler passed the TF certification](https://www.kaggle.com/questions-and-answers/196276) :bulb:

## Courses
Note on courses, you want to focus on implementing and wrtiting as many models as possible with TensorFlow. The couses below were the ones I found to have the most hands on content.

1. [Udacity Intro To Tensorflow](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
    Good if you like structure and are starting with no knowledge. 
2. [Coursera Tensorflow Developer Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)
    Everyone should take this course. This was my primary reference for study. :key:
3. [DataCamp Introduction to Deep Learning](https://www.datacamp.com/courses/introduction-to-deep-learning-with-keras)
    Good for getting started and building muscle memory with the Keras API.
4. [DataCamp Image Processing With Keras](https://www.datacamp.com/courses/image-processing-with-keras-in-python)
    Good overview of simple image models. :bulb:
5. [DataCamp Advanced Deep Learning With Keras](https://www.datacamp.com/courses/advanced-deep-learning-with-keras)
    Shows you how to use uncommon architectures in Keras. Found this highly useful for rounding out my knowledge of the Keras API. :bulb:
    
    
## Skills Checklist & TensorFlow Examples
The following summarizes the major topic areas of the exam including core skills, example notebooks, and datasets to practice on. If you find any of these notebooks please upvote and support the authors.

### 1. Tensorflow Basics
***Core Skills***
- Building, compiling, training, evaluating models for binary and muilti- classification
- Identifying and mitigating overfitting
- Plotting loss and accuracy
- Matching input and output shapes
- Using early stopping callbacks
- Using datasets from tensorflow datasets

***Example Notebooks***
1. [TensorFlow guide to getting started: TF Record & Classification](https://www.kaggle.com/nicholasjhana/tensorflow-guide-to-getting-started)
2. [TensorFlow Basics and Fundamentals](https://www.kaggle.com/rohandx1996/tensorflow-basics-and-fundamentals)
3. [TF Record Basics](https://www.kaggle.com/ryanholbrook/tfrecords-basics)

***Study Datasets***
1. [TensorFlow Dataset Catalog](https://www.tensorflow.org/datasets/catalog/overview)
2. [Mercedes Benz Manufacturign: Regression Task](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/data)
3. [Fraud Detection: Classification Task](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)
4. [Oranges vs. Grapefruit](https://www.kaggle.com/joshmcadams/oranges-vs-grapefruit)

### 2. Image Classification
***Core Skills***
- Binary- and multi-classification tasks
- Using Conv2D and MaxPooling Layers
- Understanding how convolutions improve image the nerual network
- Using image augmentation and improving overfitting
- Use ImageDataGenerator and the directory labelling structure
- Using transfer learning and model checkpoints

***Example Notebooks***
1. [Tensorflow Multi Classification Transfer Learning](https://www.kaggle.com/nicholasjhana/tensorflow-multi-classification-transfer-learning)
2. [Basic Image Classification: Tensorflow and Keras](https://www.kaggle.com/mansi75/basic-image-classification-tensorflow-and-keras)
3. [Plant Pathology 2020 : EDA + Models](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models#Modeling-)
4. [Cassava Leaf Disease - Starter Notebook](https://www.kaggle.com/lavanyask/cassava-leaf-disease-starter-notebook)
5. [Gettting Started with 100 Flowers on TPU](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu)

***Study Datasets***
1. [Plant Pathology Multi Classification](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview)
2. [Pokemon Image Dataset](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)
3. [Cats Vs. Dogs Image Dataset](https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset)
4. [Rock-Paper-Scissors Images](https://www.kaggle.com/drgfreeman/rockpaperscissors)

### 3. Natural Language Processing
***Core Skills***
- Prepare text for use in a TensorFlow model
- Use TensorFlow to identify text in binary and multi-class categorization.
- Use RNNs, LSTMs, and GRUs
- Train word embeddings and import word embedding weights
- Train LSTMs on existing text and generate text

***Example Notebooks***
1. [TensorFlow: Binary Classification on Text](https://www.kaggle.com/nicholasjhana/tensorflow-binary-classification-on-text)
2. [https://www.kaggle.com/foucardm/tensorflow-certification-guide-text-data](https://www.kaggle.com/foucardm/tensorflow-certification-guide-text-data)
3. [Keras Starter Script with Word Embeddings](https://www.kaggle.com/sudalairajkumar/keras-starter-script-with-word-embeddings)
4. [Using Word Embeddings for Data Augmentation](https://www.kaggle.com/theoviel/using-word-embeddings-for-data-augmentation)
5. [Reuter News Exploratory Data Analysis (Word Embeddings and Sentiment Analysis)](https://www.kaggle.com/hoonkeng/deep-eda-word-embeddings-sentiment-analysis)
6. [Text2Score >> keras RNN + Word Embedding](https://www.kaggle.com/gidutz/text2score-keras-rnn-word-embedding)

***Study Datasets***
1. [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction)
2. [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
3. [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
4. [What's Cooking?](https://www.kaggle.com/c/whats-cooking-kernels-only/overview/evaluation)
5. [Tensorflow Official Text Datasets](https://www.kaggle.com/imoore/tensorflow-official-text-datasets)

### 4. Timeseries Forecasting
***Core Skills***
- Using RNNs and CNNs in forecasting.
- Identify when to use trailing and centred windows
- Adjusting the learning rate with a Learning Rate Scheduler
- Preparing features and labels
- Identify and compensate Sequence Bias

***Example Notebooks***
1. [Multi-variate timeseries forecasting in TensorFlow](https://www.kaggle.com/nicholasjhana/multi-variate-time-series-forecasting-tensorflow)
2. [TensorFlow certification timeseries](https://www.kaggle.com/foucardm/tensorflow-certification-time-series)
3. [Deep Learning for Time Series Forecasting](https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting)
4. [Univariate timeseries forecasting with keras](https://www.kaggle.com/nicholasjhana/univariate-time-series-forecasting-with-keras)
5. [Time-series forecasting with LSTM autoencoders](https://www.kaggle.com/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders)

***Study Datasets***
1. [Temperature Time-Series for some Brazilian cities](https://www.kaggle.com/volpatto/temperature-timeseries-for-some-brazilian-cities)
2. [Population Time Series Data](https://www.kaggle.com/census/population-time-series-data)
3. [Daily Climate time series data](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data)
4. [Hourly energy demand generation and weather](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather)

## TensorFlow Inputs and Outputs Table
This table summarises the majority of input and model configurations for each type of classification/regression task. Helpful as a quick reference while working on problems. Original version [here](https://www.kaggle.com/nicholasjhana/tensorflow-guide-to-getting-started/edit/run/46427918).

***Inputs***

| Data Type | Input Shape |
| --- | --- |
| Image | (image height, image width, number of channels) |
| Sequence | (number of sequence steps, number of features) |
| Structured | (samples/batch size, features) |

***Outputs***

| Problem Type | Output Neurons | Target Format | Loss Type | Last Neuron Activation |
| --- | --- | --- | --- | --- |
| Binary Classification | 1 | Binary | binary_crossentropy |sigmoid|
| Multi Classification | Number of classes | One-Hot Encoded | categorical_crossentropy |softmax|
| Multi Classification | Number of classes | Label Encoded | sparse_categorical_crossentropy|softmax|
| Regression | Number of predictions | Numeric | Any regression metric: MSE/RMSE/MSLE/Huber |None|

