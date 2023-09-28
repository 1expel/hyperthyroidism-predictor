[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/YCTbQ0qx)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10595811&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

This repo contains the instructions for a machine learning project.

## Introduction

This project is a machine-learning model that predicts whether a patient has a hyperthyroid condition. The project uses a dataset from the Thyroid Disease Database from UCI machine learning repository which has more than 3000 cases. The dataset has 22 variables which are medical characteristics of the patient that a machine learning model can use to assist human diagnosis.  

## Objectives

Create a machine-learning model that can assist in the human diagnosis of patients with a hyperthyroid condition.

## Methodology

Process the Data: cleaning the data, feature selection, ensuring proper data types, and fixing invalid values, columns, tables, etc.
Visualize the Data: visualizing the data helps with interpreting it. data visualization techniques such as scatter plots, histograms, and box plots can be used to gain a logical understanding of the data and the relationship between variables in the data. 
Train and Optimize the Models: using various machine learning models such as random forests and knn, we will train the models on the data, and optimize them using hyperparam tuning. 
Performance Metrics: we will use several performance metrics to evaluate our models on test data. some of the metrics we will use include accuracy, precision, recall, specificity, and f1 score. using these performance metrics we can pick the best model we can use on new unseen data in real life use.

## Running the Project

The project is run through main.py located in the src folder.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data     <- downloads the data, processes the data, and builds features      
       │   └── make_dataset.py 
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
       │   └── visualize.py
       │  
       └── main.py <- Runs the code
