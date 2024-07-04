__INTRODUCTION__

This repository contains two data science projects:<br/>

--> Iris Species Classification<br/>
--> Titanic Survival Prediction<br/>
Each task includes data preprocessing, model training, and evaluation steps to solve the respective classification problems.<br/>

*__1. IRIS FLOWER CLASSIFICATION__*

_Dataset_
The dataset used for this project is the famous Iris dataset, which can be found at the UCI Machine Learning Repository: [dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

_Project Overview_
The Iris classification project involves predicting the species of an Iris flower (setosa, versicolor, or virginica) based on its sepal length, sepal width, petal length, and petal width. Various machine learning models are trained and evaluated for this purpose.

_Steps_
Data Loading and Exploration: Load the dataset and explore its structure.<br/>
Data Preprocessing: Encode the target variable (species) and handle any missing values.<br/>
Data Visualization: Visualize the distribution of species and feature correlations.<br/>
Model Training: Train multiple models including Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine.<br/>
Model Evaluation: Evaluate the models using accuracy scores on the training and test sets.<br/>
Prediction Interface: Create a Gradio interface to input flower measurements and predict the species using the trained models.<br/>

_How to Run_
//pip install pandas numpy seaborn scikit-learn gradio<br/>
//python iris_classification.py<br/>


_Code File_
iris_classification.py: Contains the code for loading the dataset, training the models, and creating the Gradio interface.<br/>


*__2. TITANIC SURVIVAL PREDICTION__*

_Dataset_
The dataset used for this project is the Titanic dataset, which can be found at Kaggle: [dataset](https://www.kaggle.com/datasets/ashishkumarjayswal/titanic-datasets).<br/>

_Project Overview_
The Titanic survival prediction project involves predicting whether a passenger survived or not based on various features such as age, sex, passenger class, and more. A Support Vector Machine (SVM) model is trained and evaluated for this purpose.<br/>

_Steps_
Data Loading and Exploration: Load the dataset and explore its structure.<br/>
Data Preprocessing: Handle missing values, encode categorical variables, and scale numerical features.<br/>
Data Visualization: Visualize the distribution of survival and feature relationships.<br/>
Model Training: Train a Support Vector Machine (SVM) model.<br/>
Model Evaluation: Evaluate the model using classification metrics such as accuracy, confusion matrix, and classification report.<br/>

_How to Run_
//pip install pandas numpy matplotlib seaborn scikit-learn<br/>

Run the survival_prediction.ipynb notebook to execute the steps in the project.<br/>

_Code File_
survival_prediction.ipynb: Contains the code for loading the dataset, preprocessing the data, training the SVM model, and evaluating the model.<br/>

_Conclusion_
These projects demonstrate the application of machine learning techniques to solve classification problems using well-known datasets. The Iris classification project includes a user-friendly Gradio interface for making predictions, while the Titanic survival prediction project provides a detailed notebook for data exploration and model evaluation.<br/>
