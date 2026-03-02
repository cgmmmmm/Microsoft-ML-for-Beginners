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

## 1. Computational Thinking
Before writing any code, we must understand how to frame a problem for a machine. This section covers the **Machine Learning Paradigm**: how we shift from traditional programming (giving a computer rules to produce answers) to machine learning (giving a computer data and answers to infer the rules). 

<br>

## 3. Unlabeled and Labeled Data

Machine learning fundamentally depends on the structure of data.

You will learn:
* What labeled data is.
* What unlabeled data is.
* Why the presence or absence of labels determines the learning strategy.

<br>

## 3. Supervised and Unsupervised Learning
This section breaks down the two primary approaches to machine learning:
* **Supervised Learning:** Training a model using "labeled data" (where the correct answer is provided to the machine during training).
* **Unsupervised Learning:** Allowing the model to find hidden patterns in "unlabeled data" without any predefined answers.

We examine when and why each approach is used.

<br>

## 4. Feature Engineering
Raw data is rarely usable in its original form.

How do we describe a real-world object to an algorithm?

This section explores:
* How to select and extract relevant characteristics (**features**) from your data.
* How to represent examples as feature vectors, a critical step that dictates how well your model will eventually perform.
* The impact of feature design on model performance.

This is one of the most critical steps in Machine Learning.

<br>

## 5. The Minkowski Metric
Once data is represented numerically, we require a mathematical definition of similarity.

We need a mathematical way to determine how "similar" or "different" two examples are. 

This section introduces:
* The **Minkowski distance**, a generalized metric for calculating distance between data points in space.
* You will see how common measurements like **Manhattan distance (L1 Norm)** and **Euclidean distance (L2 Norm)** are just specific variations of the Minkowski metric.
* The impact of scaling and feature weighting.

You will understand how distance metrics influence clustering and classification outcomes.

<br>

## 6. Clustering and Classification
Building on the previous section, we dive into the most common tasks associated with those paradigms:
* **Classification:** A supervised method used to predict discrete categories (e.g., determining if an email is "Spam" or "Not Spam").
* **Clustering:** An unsupervised method used to group inherently similar data points together into natural clusters.

You will see how feature engineering and distance metrics directly shape these tasks.