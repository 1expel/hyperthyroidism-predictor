import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train():

    print("\nREADING TRAIN DATA...\n")

    # READ PROCESSED DATA FILES
    train_data = pd.read_csv('data/processed/train_data.csv')
    
    # DIVIDE PROCESSED TRAIN DATA INTO X AND Y
    X_train = train_data.iloc[:, 0:37]
    y_train = train_data.iloc[:, 37:]

    print("\n----- DATA SPLIT -----\n")

    print("\nX FEATURES...\n")

    print(X_train.info())

    print("\nY CLASSES...\n")

    print(y_train.info())

    print("\nTRAINING MODELS...\n")

    # HYPERTUNING/TRAINING DescisionTreeClassifier
    param_grid = {
        "max_depth": range(1,11)
    }
    grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=10)
    grid_search_dt.fit(X_train, y_train)
    best_params = grid_search_dt.best_params_
    fit_dt = grid_search_dt.best_estimator_
    best_score = grid_search_dt.best_score_
    # fit_dt = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)

    print("knn hyperparam tuning results:")
    print("best params: ", best_params)
    print()

    # HYPERTUNING/TRAINING KNeighborsClassifier
    param_grid = {
        "n_neighbors": range(1,11,2)
    }
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10)
    grid_search_knn.fit(X_train, y_train)
    best_params = grid_search_knn.best_params_
    fit_knn = grid_search_knn.best_estimator_
    best_score = grid_search_knn.best_score_
    # fit_knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)

    print("decision tree hyperparam tuning results:")
    print("best params: ", best_params)

    return fit_dt, fit_knn
