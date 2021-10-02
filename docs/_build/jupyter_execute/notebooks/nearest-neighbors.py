#!/usr/bin/env python
# coding: utf-8

# # Nearest Neighbors

# ```{admonition} Attribution
# This notebook is based on [Lecture 2](https://github.com/rasbt/stat451-machine-learning-fs20/tree/master/L02) of [Sebastian Raschka](http://pages.stat.wisc.edu/~sraschka/)'s [STAT 451: Intro to Machine Learning @ UW-Madison (Fall 2020)](http://pages.stat.wisc.edu/~sraschka/teaching/stat451-fs2020/).
# ```

# ## Introduction

# While KNN is a universal function approximator under certain conditions, the underlying
# concept is relatively simple. 

# 1. lazy learner, i.e. all processing of training examples occurs during inference. training = just store data.
# 2. computing m(q) on a trained knn model m just looks for the k nearest neighbors of a query point q and uses properties of those k train examples to compute a class label &mdash; or a continuous target, in the case of regression.
# 
# the overall idea is that instead of approximating a function globally during a prediction, m approximates the target function locally.
# 
# put image knn-query.png
# 
# Illustration of the nearest neighbor classification algorithm in two dimensions (features
# x1 and x2). In the left subpanel, the training examples are shown as blue dots, and a query point
# that we want to classify is shown as a question mark. In the right subpanel, the class labels are
# annotated via blue squares and red triangle symbols. The dashed line indicates the nearest neighbor
# of the query point, assuming a Euclidean distance metric. The predicted class label is the class
# label of the closest data point in the training set (here: class 0, i.e., blue square).

# ### Nearest neighbors in context

# Since the prediction is based on a
# comparison of a query point with data points in the training set (rather than a global
# model), kNN is also categorized as instance-based (or “memory-based”) method. In contrast, SVM is an eager instance-based learning algorithm. Lastly, because we do not make any assumption about the functional form of the kNN
# algorithm, a kNN model is also considered a nonparametric model.
# 
# 

# Under certain assumptions, we can estimate the conditional probability that a given data
# point belongs to a given class as well as the marginal probability for a feature given a training
# dataset (more details are provided in the section on “kNN from a Bayesian Perspective”
# later). However, since kNN does not explicitly try to model the data generating process
# but models the posterior probabilities, p(f(x) = i|x), directly, kNN is usually considered a
# discriminative model.

# ### Common use cases

# While neural networks are gaining popularity in the computer vision and pattern recognition
# field, one area where k-nearest neighbors models are still commonly and successfully being
# used is in the intersection between computer vision, pattern classification, and biometrics
# (e.g., to make predictions based on extracted geometrical features2).
# Other common use cases include recommender systems (via collaborative filtering3) and
# outlier detection4.
# 

# ## Nearest Neighbors Algorithm

# In[17]:


import numpy as np


class NearestNeighbor:
    """Implementing the Nearest Neighbor algorithm."""

    def __init__(self, metric="Euclidean"):
        if metric == "Euclidean":
            self.distance = lambda x, y: np.sqrt(((x - y)**2).sum())
        else:
            raise NotImplementedError

    def fit(self, X, y):
        """Fit the model. Here X is a 2d numpy array and y is array-like."""

        self.data = X
        self.targets = y

    def predict(self, query_point):
        """Predict label of query point."""

        query_point = np.array(query_point)
        t_min = None
        min_distance = np.inf
        
        for i in range(len(self.data)):
            x = self.data[i]
            t = self.targets[i]
            distance = self.distance(query_point, x)
            if distance < min_distance:
                min_distance = distance
                t_min = t

        return t_min


# Testing with a simple dataset and query point.

# In[18]:


# Generate fake dataset
X = np.array([
    [1, 2],
    [3, 3],
    [5, 1]
])
y = [0, 1, 2]

# Initialize NN model
model = NearestNeighbor()

# Fit and make inference
model.fit(X, y)
print(model.predict([1, 1]))


# The default distance for KNN algorithm is the Euclidean distance
# 
# $$d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_{j=1}^m {\left(x_j - y_j\right)}^2 }.$$
# 
# 

# ## Nearest Neighbor Decision Boundary

# 
