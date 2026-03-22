# Chapter 1: Introduction to Machine Learning

Welcome to Chapter 1! This module serves as your foundational entry point into the world of Machine Learning (ML). 

<br>

By the end of this chapter, you will understand:
* The core paradigms of ML
* How to think computationally about real-world problems
* The distinction between labeled and unlabeled data
* The difference between supervised and unsupervised paradigms
* How to engineer meaningful features
* How to measure similarity using mathematical distance metrics
* How clustering and classification arise from these principles
* How to represent data so a computer can learn from it

<br>

### How to Use This Chapter
Navigate through the folders in numerical order. Each folder contains a `notes.md` file with the core lecture concepts, examples, and formulas required to master the topic.

<br>

## 1.1. Computational Thinking
Before writing any code, we must understand how to frame a problem for a machine. This section covers the **Machine Learning Paradigm**: how we shift from traditional programming (giving a computer rules to produce answers) to machine learning (giving a computer data and answers to infer the rules). 

<br>

## 1.2. Unlabeled and Labeled Data

Machine learning fundamentally depends on the structure of data.

You will learn:
* What labeled data is.
* What unlabeled data is.
* Why the presence or absence of labels determines the learning strategy.

<br>

## 1.3. Supervised and Unsupervised Learning
This section breaks down the two primary approaches to machine learning:
* **Supervised Learning:** Training a model using "labeled data" (where the correct answer is provided to the machine during training).
* **Unsupervised Learning:** Allowing the model to find hidden patterns in "unlabeled data" without any predefined answers.

We examine when and why each approach is used.

<br>

## 1.4. Feature Engineering
Raw data is rarely usable in its original form.

How do we describe a real-world object to an algorithm?

This section explores:
* How to select and extract relevant characteristics (**features**) from your data.
* How to represent examples as feature vectors, a critical step that dictates how well your model will eventually perform.
* The impact of feature design on model performance.

This is one of the most critical steps in Machine Learning.

<br>

## 1.5. The Minkowski Metric
Once data is represented numerically, we require a mathematical definition of similarity.

We need a mathematical way to determine how "similar" or "different" two examples are. 

This section introduces:
* The **Minkowski distance**, a generalized metric for calculating distance between data points in space.
* You will see how common measurements like **Manhattan distance (L1 Norm)** and **Euclidean distance (L2 Norm)** are just specific variations of the Minkowski metric.
* The impact of scaling and feature weighting.

You will understand how distance metrics influence clustering and classification outcomes.

<br>

## 1.6. Techniques of ML & The Lifecycle of a ML Model
Building a Machine Learning model from scratch is a complex process that takes time and resources. It is not just about writing algorithms; it requires a structured, end-to-end approach.

This section explores the complete journey of an ML project from conception to production:

1. **Problem Framing**: Determining if ML is actually the right approach, or if traditional programming would be more efficient.

2. **Data Preparation**: Collecting, cleaning, and transforming raw data so a machine can understand it.

3. **Training**: Teaching your chosen algorithm to recognize patterns using historical data.

4. **Evaluation**: Testing the model on unseen data to ensure it generalizes well and isn't just memorizing answers.

5. **Hyperparameter Tuning**: Systematically adjusting the algorithm's settings to fix underfitting or overfitting.

6. **Deployment**: Integrating the trained model into the real world and monitoring it for data drift.

You will understand the step-by-step pipeline that data scientists use to build reliable and effective AI solutions.