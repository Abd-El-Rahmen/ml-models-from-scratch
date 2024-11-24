from sklearn.metrics import mean_squared_error , r2_score


def evaluate_model(model, X_test , Y_test):
    Y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_true=Y_test,y_pred=Y_pred )
    r2 = r2_score(Y_test , Y_pred)

    return rmse , r2