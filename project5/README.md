# Arvato Customer Segmentation and Predictive Modeling

A Udacity project aimed at creating customer segmentation for Arvato Bertelsmann and developing a predictive model to identify potential new customers.

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Overview](#technical-overview)
- [Requirements](#requirements)

***

<a id='project-overview'></a>

## 1. Project Overview

This project involves analyzing demographic data from a mail-order company in Germany alongside general population data from Germany. The objective is to identify potential new customers for the company.

The project is divided into three phases:
1. **Data Processing**: Process the demographic data of the general population and the customers of Arvato.
2. **Customer Segmentation**: Utilize customer segmentation techniques to further analyze and process the data, preparing it for predictive modeling.
3. **Predictive Modeling**: Employ supervised learning methods to build a model that predicts the likelihood of an individual becoming a customer.

The main goal is to predict which individuals are most likely to become customers for a German mail-order sales company.

<a id='technical-overview'></a>

## 2. Technical Overview

The workflow follows a structured approach from data exploration and processing to inference. Given the large volume of data, we build a preprocessing pipeline to eliminate unnecessary and outlier data, and implement dimensionality reduction and clustering techniques to identify segments. Due to the nature of the data, AUC/ROC is used as the evaluation metric. Predictions for the test set are submitted to a Kaggle competition for evaluation.

Key concepts covered and implemented in the project include:
- Data Exploration & Cleaning
- Dimensionality Reduction
- Clustering
- Supervised Learning
- Final Model Evaluation
- Feature Importance Analysis
- Relevance Analysis of Important Features in Clusters

<a id='requirements'></a>

## 3. Requirements

- Pandas
- Scikit-learn (Sklearn)
- XGBoost
