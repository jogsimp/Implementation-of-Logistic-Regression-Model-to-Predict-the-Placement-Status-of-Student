# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and separate the input features (CGPA, Internship, Communication_Score) and the target label (Placed).
2. Split the data into training and testing sets, then apply standard scaling to normalize the feature values.
3. Train a Logistic Regression model using the scaled training dataset.
4. Predict placement status and probabilities for the test data, and display the actual vs. predicted results.



## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Placement_Data.csv")

X = df.drop(["status", "salary"], axis=1)
y = (df["status"] == "Placed").astype(int)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)


y_pred = np.array(y_pred) 

result_df = pd.DataFrame({
    "Student_ID": range(1, len(y_pred) + 1),
    "Predicted_Profit": np.round(y_pred, 2)
})

print(result_df.to_string(index=False))


Developed by: Joshua Abraham Philip A
RegisterNumber: 25013744
*/
```

## Output:
<img width="338" height="595" alt="Screenshot 2025-12-04 at 9 57 29 PM" src="https://github.com/user-attachments/assets/896af73f-ea95-4766-a514-f3b06810714e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
