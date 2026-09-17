import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def main():

    # Prepare and load the dataset
    data = pd.read_csv("HepatitisCdata.csv", index_col=0)
    data["Sex"] = data["Sex"].map({"m": 0, "f": 1})

    # Define the classification target and features
    target = data["Category"] 
    features = data.drop("Category", axis=1)
    class_names = ["Donor", "Suspected Donor", "Hepatitis", "Fibrosis", "Cirrhosis"]

    # Split the shuffled data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)

    # Train the model using random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    # Use the model to make predictions on the test data
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # Use the predictions on the test set to determine the model's accuracy
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"ROC-AUC Score (OVR, Weighted): {roc_auc_score(y_test, y_probs, multi_class='ovr', average='weighted'):.4f}\n")

    # Illustrate the model's performance with a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.show()

    # Obtain data for a new individual
    print("Data for new individual:")
    individual = pd.DataFrame([[
        float(input("Age: ")), input("Sex (m/f): "),
        float(input("ALB: ")), float(input("ALP: ")),
        float(input("ALT: ")), float(input("AST: ")),
        float(input("BIL: ")), float(input("CHE: "))]],
        columns = X_train.columns)
    
    individual["Sex"] = individual["Sex"].map({"m": 0, "f": 1})

    # Predict whether the new individual is a patient or blood donor
    prediction = model.predict(individual)[0].split("=")[1]

    print(f"Predicted Category: {prediction}")

if __name__ == "__main__":
    main()