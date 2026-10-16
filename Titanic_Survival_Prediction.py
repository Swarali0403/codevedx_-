
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# Load the dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select input columns
X = data[["Pclass", "Sex", "Age", "SibSp",
          "Parch", "Fare", "Embarked"]].copy()
y = data["Survived"]

# Handle missing values
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])


# Charts
plt.figure(figsize=(6, 4))
data["Survived"].value_counts().sort_index().plot(
    kind="bar", color=["tomato", "mediumseagreen"]
)
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Survival by gender
gender_survival = data.groupby("Sex")["Survived"].mean() * 100

plt.figure(figsize=(6, 4))
gender_survival.plot(kind="bar", color=["cornflowerblue", "orange"])
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Survival by passenger class
class_survival = data.groupby("Pclass")["Survived"].mean() * 100

plt.figure(figsize=(6, 4))
class_survival.plot(kind="bar", color=["gold", "mediumseagreen", "salmon"])
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Age distribution of survivors and non-survivors
plt.figure(figsize=(7, 4))
plt.hist(
    data.loc[data["Survived"] == 0, "Age"].dropna(),
    bins=20, alpha=0.6, color="tomato", label="Did Not Survive"
)
plt.hist(
    data.loc[data["Survived"] == 1, "Age"].dropna(),
    bins=20, alpha=0.6, color="royalblue", label="Survived"
)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.legend()
plt.tight_layout()
plt.show()

# Convert text values into numbers
X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed!")

# Predict survival
y_pred = model.predict(X_test)

print("\nModel Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report in percentages
report = classification_report(
    y_test, y_pred,
    target_names=["Did Not Survive", "Survived"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

# Convert precision, recall, and F1-score to percentages
report_df[["precision", "recall", "f1-score"]] *= 100

print("\nClassification Report (Percentages):")
print(report_df.round(2).to_string())

# Predict for a new passenger
new_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "female",
    "Age": 26,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.92,
    "Embarked": "S"
}])

new_passenger = pd.get_dummies(
    new_passenger, columns=["Sex", "Embarked"], dtype=int
)
new_passenger = new_passenger.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(new_passenger)[0]

print("\nNew Passenger Prediction:")
if prediction == 1:
    print("Passenger is predicted to SURVIVE.")
else:
    print("Passenger is predicted NOT TO SURVIVE.")
