import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv")

# Features and labels
X = df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Set MLflow experiment
mlflow.set_experiment("Email_Spam_Detection")

# Models to compare
models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression()
}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train_vec, y_train)

        # Predict
        predictions = model.predict(X_test_vec)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model_name", name)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, name=name)

        print(f"{name} Accuracy:", accuracy)