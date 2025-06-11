def test_linear_regression_fit_predict_score():
    import numpy as np
    from ml_models.linear_regression import LinearRegression

    # Dati fittizi
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + 0.1 * np.random.randn(100, 1)  # Rumore standard

    # Test con formula chiusa
    print("\n--- Test with closed formula ---")
    model_closed = LinearRegression()
    model_closed.fit(X, y)

    predictions_closed = model_closed.predict(X)
    print("Coefficients (self.coef_):", model_closed.coef_.flatten())
    print("First 5 predictions:", predictions_closed[:5].flatten())
    print("First 5 y reali:", y[:5].flatten())
    r2_closed = model_closed.score(X, y)
    print("R² score (closed formula):", r2_closed)

    assert predictions_closed.shape == y.shape
    assert r2_closed > 0.9

    # Test con gradient descent
    
    # Dati fittizi
   
 # Test con gradient descent

    np.random.seed(50)
    X = 2 * np.random.rand(100000, 1)
    y = 4 + 3 * X + 0.1 * np.random.randn(100000, 1)  # Aggiungi un po' di rumore per rendere i dati realistici


    print("\n--- Test with gradient descent ---")
    model_gd = LinearRegression()
    model_gd.fit(X, y, use_gradient_descent=True)

    predictions_gd = model_gd.predict(X)
    print("Coefficients (self.theta):", model_gd.theta.flatten())
    print("First 5 predictions:", predictions_gd[:5].flatten())
    print("First 5 y reals:", y[:5].flatten())
    r2_gd = model_gd.score(X, y)
    print("R² score (gradient descent):", r2_gd)

    assert predictions_gd.shape == y.shape
    assert r2_gd > 0.9

