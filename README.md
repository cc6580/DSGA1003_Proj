# DSGA1003_Proj: Fake Review Detection

## Dataset

This project uses datasets from [Yelp](https://worksheets.codalab.org/worksheets/0x33171fbfe67049fd9b0d61962c1d05ff).
This dataset includes reviews for restaurants located in New York City. Reviews include product and `user id`, `timestamp`, `ratings`, and a plaintext `review`. Yelp has a filtering algorithm in place that identifies fake/suspicious reviews and separates them into a filtered list. This Yelp dataset contains both recommended and filtered reviews. We consider them as genuine and fake, respectively. The positive classes (+1) are fake reviews and the negative classes are genuine reviews (0). Note that the classes are imbalanced, with around 10% fake reviews.

## Research Objectives
The project goal is to predict whether a review is fake or not, i.e. a binary classification task. Two evaluation metrics are used, [auROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) and [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score).

## Proposed ml models
1. Naive Bayes
2. Logistic Regression
3. SVM
4. Decision tree / random forest / xgboost
5. NN
6. Novel Algorithm
