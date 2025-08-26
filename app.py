from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import train_model, predict
from evaluate import evaluate_model

def main():
    # Load dataset
    df = load_data()

    # Features and Target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Predictions
    y_pred = predict(model, X_test)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, y_pred)

    # Print results
    print(f"Mean Squared Error: {results['Mean Squared Error']}")
    print(f"R2 Score: {results['R2 Score']}")
    print(f"Intercept: {results['Intercept']}")
    print("\nCoefficient for each feature:")
    print(results["Coefficients"])

if __name__ == "__main__":
    main()
