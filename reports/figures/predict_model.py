import pandas as pd

def predict(fit_dt, fit_knn):

    print("\nREADING TEST DATA...\n")

    # READ PROCESSED DATA FILES
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # DIVIDE PROCESSED TEST DATA INTO X AND Y
    X_test = test_data.iloc[:, 0:37]
    y_test = test_data.iloc[:, 37:]

    print("\nMAKING PREDICITONS...\n")

    # DescisionTreeClassifier PREDICT
    dtree_pred = fit_dt.predict(X_test)

    # KNeighborsClassifier PREDICT
    pred_knn = fit_knn.predict(X_test)

    return y_test, dtree_pred, pred_knn
