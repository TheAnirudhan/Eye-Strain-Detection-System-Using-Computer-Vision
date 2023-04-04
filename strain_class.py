import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# Input data: Features and Labels
# Features: screen time, distance from screen, brightness, and break frequency

# Load and preprocess the data
data = pd.read_csv("eye_strain_data_1.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Labels: Eye strain score (0: Low, 1: Moderate, 2: High)


# Split the data into training and testing sets
'''
# Train the SVM model
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)


# Create a logistic regression model and train it

clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
'''
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the model to the training data
clf.fit(X_train, y_train)
# Predict the eye strain score
y_pred = clf.predict(X_test)

# Calculate the accuracy of the prediction
#accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print("\n\n")
print(y_test)


# calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# print the results
print(f"AUC score: {roc_auc:.2f}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
cf_matrix=confusion_matrix(y_test, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()
'''
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
'''