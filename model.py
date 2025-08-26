from sklearn.linear_model import LinearRegression

def train_model(x_train, y_train):
    """Train and return Linear Regression model."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    """Generate predictions using trained model."""
    return model.predict(x_test)
