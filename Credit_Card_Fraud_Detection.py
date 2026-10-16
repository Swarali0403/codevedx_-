
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("creditcard.csv")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTransaction counts:")
print(df["Class"].value_counts())

print("\nTransaction percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(2))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFraud and genuine transaction details:")
print(df.groupby("Class")["Amount"].describe())

# Remove missing values
df = df.dropna()

# Separate input features and target
x = df.drop("Class", axis=1)
y = df["Class"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Time and Amount
scaler = StandardScaler()

x_train = x_train.copy()
x_test = x_test.copy()

x_train[["Time", "Amount"]] = scaler.fit_transform(
    x_train[["Time", "Amount"]]
)

x_test[["Time", "Amount"]] = scaler.transform(
    x_test[["Time", "Amount"]]
)

print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Create machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

results = []

# Train and evaluate the models
for name, model in models.items():

    model.fit(x_train, y_train)

    pred = model.predict(x_test)


    print("Model:", name)

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=["Genuine", "Fraud"],
        zero_division=0
    ))

    # Calculate evaluation scores
    accuracy = accuracy_score(y_test, pred) * 100
    precision = precision_score(y_test, pred, zero_division=0) * 100
    recall = recall_score(y_test, pred, zero_division=0) * 100
    f1 = f1_score(y_test, pred, zero_division=0) * 100

    print("Accuracy:", round(accuracy, 2), "%")
    print("Fraud Precision:", round(precision, 2), "%")
    print("Fraud Recall:", round(recall, 2), "%")
    print("Fraud F1 Score:", round(f1, 2), "%")

    results.append([
        name, accuracy, precision, recall, f1
    ])

    # Display confusion matrix
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Genuine", "Fraud"]
    )

    display.plot(cmap="Purples", values_format="d")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Compare both models
result = pd.DataFrame(
    results,
    columns=[
        "Model", "Accuracy", "Precision", "Recall", "F1 Score"
    ]
)


print("MODEL COMPARISON (PERCENTAGE)")
print(result.round(2).to_string(index=False))

# Plot model comparison
result.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6),
    color=["orange", "seagreen", "mediumpurple", "cornflowerblue"]
)

plt.title("Credit Card Fraud Detection - Model Comparison")
plt.ylabel("Percentage (%)")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Display transaction distribution
df["Class"].value_counts().rename(
    index={0: "Genuine", 1: "Fraud"}
).plot(
    kind="bar",
    color=["seagreen", "tomato"],
    figsize=(7, 5)
)

plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


