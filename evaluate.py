import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, x_test, y_test, y_pred):
    """Evaluate the model and return performance metrics + coefficients DataFrame."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    coef_df = pd.DataFrame(model.coef_, x_test.columns, columns=['Coefficient'])

    results = {
        "Mean Squared Error": mse,
        "R2 Score": r2,
        "Intercept": model.intercept_,
        "Coefficients": coef_df
    }
    return results
