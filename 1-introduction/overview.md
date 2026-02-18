# Chapter 1: Introduction to Machine Learning

Welcome to Chapter 1! This module serves as your foundational entry point into the world of Machine Learning (ML). 
<br><br>
By the end of this chapter, you will understand the core paradigms of ML, how to represent data so a computer can learn from it, and how to measure similarities between different data points.

### How to Use This Chapter
Navigate through the folders in numerical order. Each folder contains a `notes.md` file with the core lecture concepts, examples, and formulas required to master the topic.



## 1. Computational Thinking
Before writing any code, we must understand how to frame a problem for a machine. This section covers the **Machine Learning Paradigm**: how we shift from traditional programming (giving a computer rules to produce answers) to machine learning (giving a computer data and answers to infer the rules). 

## 2. Feature Engineering
How do we describe a real-world object to an algorithm? This section explores how to select and extract relevant characteristics (**features**) from your data. You will learn how to represent examples as feature vectors, a critical step that dictates how well your model will eventually perform.

## 3. The Minkowski Metric


Once we have our features, we need a mathematical way to determine how "similar" or "different" two examples are. This section introduces the **Minkowski distance**, a generalized metric for calculating distance between data points in space. You will see how common measurements like **Manhattan distance** and **Euclidean distance** are just specific variations of the Minkowski metric.

## 4. Supervised and Unsupervised Learning
This section breaks down the two primary approaches to machine learning:
* **Supervised Learning:** Training a model using "labeled data" (where the correct answer is provided to the machine during training).
* **Unsupervised Learning:** Allowing the model to find hidden patterns in "unlabeled data" without any predefined answers.

## 5. Clustering and Classification
Building on the previous section, we dive into the most common tasks associated with those paradigms:
* **Classification:** A supervised method used to predict discrete categories (e.g., determining if an email is "Spam" or "Not Spam").
* **Clustering:** An unsupervised method used to group inherently similar data points together into natural clusters.