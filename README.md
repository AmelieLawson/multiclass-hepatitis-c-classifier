# Multi-class Hepatitis C Classifier

Adapting my first machine learning project from a binary to a multi-class classification problem.

## Project Overview

The aim of this project was to use machine learning to classify individuals based on their age, sex, and various laboratory values.

The data was used to train a Random Forest Classifier model to predict whether an unknown individual was a blood donor, suspected blood donor, hepatitis C, fibrosis, or cirrhosis patient.

## Dataset

The dataset used in this project is the Hepatitis C Prediction Dataset, obtained from Kaggle. It is not included in this repository. Place `HepatitisCdata.csv` in the same directory as `hepatitis.py` before running the program.

[Hepatitis C Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset)

The data was originally obtained from UCI Machine Learning Repository. It contains laboratory values for a range of blood donors and hepatitis C patients along with demographic values like age and sex.


## Method

The steps involved in the project are outlined below:

1. Load the dataset using pandas.
2. Convert the "Sex" variable from categoric (m/f) to numerical (0/1).
3. Define the features and target variable.
4. Split the data into a training set and a test set using a 75/25 split.
5. Train a Random Forest Classifier with 100 estimators.
6. Predict the outcomes of the test set.
7. Evaluate the model using its accuracy, ROC-AUC, classification report and confusion matrix.
8. Allow the user to enter data for a new individual and generate a prediction.

I learnt this method from [Scientific Computing for Chemists with Python](https://weisscharlesj.github.io/SciCompforChemists/notebooks/introduction/intro.html#) by Charles J. Weiss.

## Evaluation

The metrics used the evaluate the model were as follows:

- Accuracy: The proportion of test predictions which were correct.
- ROC/AUC (OVR and weighted): A weighted average of how well the model distinguishes each class from all the rest combined.
- Classification report: Includes precision, recall, and F1 score for each class.
- Average macro F1 score: The average of the F1 scores with all classes treated equally.
- Average weighted F1 score: The average of the F1 scores weighted by class size.
- Confusion matrix: Illustrates the nature of the prediction errors made by the model.

## How to Run

Install the required Python libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## What I Learnt

Converting from a binary to a multi-class classification problem has taught me the following:

- How to generate a confusion matrix.
- The significance of a confusion matrix.
- The difference between an average macro and an average weighted F1 score
- The difference between OVO (one-vs-one) and OVR (one-vs-rest) ROC/AUC scores

## Additional Considerations

I chose to use the parameter `class_weight="balanced"` when training the Random Forest Classifier in order to give more importance to the particularly small suspected blood donors class. I found this tweak gave higher accuracy and average F1 scores (both macro and weighted), while only very slightly decreasing ROC/AUC.
