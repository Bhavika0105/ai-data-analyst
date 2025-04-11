import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



def train_model(df, target_column):
    """
    Trains a simple Logistic Regression model on the dataset.
    Assumes target is classification.
    """
    print("🛠 Starting model training...")

    df_numeric = df.select_dtypes(include=['int64', 'float64'])

    if target_column not in df_numeric.columns:
        print(f"❌ Target column '{target_column}' not found in numeric data!")
        return

    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained successfully!")
    print(f"🎯 Accuracy on test set: {acc:.2f}")

    # Save the model
    model_filename = "trained_model.pkl"
    joblib.dump(model, model_filename)
    print(f"💾 Model saved successfully as '{model_filename}'")
