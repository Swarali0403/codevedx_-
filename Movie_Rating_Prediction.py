import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Load the dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="utf-8", encoding_errors="replace")

# Remove extra spaces from text
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].astype("string").str.strip()

for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].astype("object")
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else np.nan
    )

# Clean column names
df.columns = df.columns.str.strip()

# Convert year, duration and votes into numbers
df["Year"] = pd.to_numeric(
    df["Year"].str.extract(r"(\d{4})")[0], errors="coerce"
)

df["Duration"] = pd.to_numeric(
    df["Duration"].str.extract(r"(\d+)")[0], errors="coerce"
)

df["Votes"] = pd.to_numeric(
    df["Votes"].str.replace(",", "", regex=False), errors="coerce"
)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Remove rows without a valid rating
df = df.dropna(subset=["Rating"])
df = df[(df["Rating"] >= 0) & (df["Rating"] <= 10)]

print("Total movies:", len(df))
print("\nFirst five movies:")
print(df.head())

# Select features for prediction
x = df[[
    "Year", "Duration", "Votes", "Genre",
    "Director", "Actor 1", "Actor 2", "Actor 3"
]]

y = df["Rating"]

# Separate number and text columns
num_cols = ["Year", "Duration", "Votes"]
text_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

# Fill missing values and prepare the data
num_data = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

text_data = Pipeline([
    ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

prepare = ColumnTransformer([
    ("numbers", num_data, num_cols),
    ("text", text_data, text_cols)
])

# Split data into training and testing parts
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create two models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=50, max_depth=15,
        random_state=42, n_jobs=-1
    )
}

scores = []
trained_models = {}

# Train and test the models
for name, model in models.items():

    movie_model = Pipeline([
        ("prepare", prepare),
        ("model", model)
    ])

    movie_model.fit(x_train, y_train)
    pred = movie_model.predict(x_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    scores.append([name, mae, mse, rmse, r2])
    trained_models[name] = movie_model

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

# Compare the results
result = pd.DataFrame(
    scores, columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"]
)

print("\nModel Comparison:")
print(result.to_string(index=False))

# Select the model with the lowest RMSE
best_name = result.loc[result["RMSE"].idxmin(), "Model"]
best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# Predict the rating of a new movie
new_movie = pd.DataFrame([{
    "Year": 2025,
    "Duration": 120,
    "Votes": 500,
    "Genre": "Drama, Romance",
    "Director": "Example Director",
    "Actor 1": "Actor A",
    "Actor 2": "Actor B",
    "Actor 3": "Actor C"
}])

rating = best_model.predict(new_movie)[0]
rating = np.clip(rating, 0, 10)

print("Predicted Movie Rating:", round(rating, 2), "/ 10")

# Show actual and predicted ratings
actual = y_test.iloc[:100]
predicted = best_model.predict(x_test.iloc[:100])

plt.figure(figsize=(8, 5))
plt.scatter(actual, predicted, color="purple")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid()
plt.tight_layout()
plt.show()

# Compare model errors
plt.figure(figsize=(7, 5))
plt.bar(result["Model"], result["RMSE"],
        color=["orange", "seagreen"])
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.title("Movie Rating Model Comparison")
plt.tight_layout()
plt.show()
