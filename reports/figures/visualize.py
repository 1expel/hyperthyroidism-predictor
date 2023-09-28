from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score, accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

def visualize(y_test, dtree_pred, pred_knn):

    print("\nVISUALIZING THE MODELS' PERFORMANCES...\n")

    # DescisionTreeClassifier PERFORMANCE
    accuracy_dt = accuracy_score(y_test, dtree_pred)
    classification_report_dt = classification_report(y_test, dtree_pred)
    print("The accuracy achieved by decision tree model:", accuracy_dt)
    print('#'*60)
    print("The classification report of decision tree model: \n", classification_report_dt)

    # KNeighborsClassifier PERFORMANCE
    accuracy_knn = accuracy_score(y_test, pred_knn)
    classification_report_knn = classification_report(y_test, pred_knn)
    print("The accuracy achieved by knn model:", accuracy_knn) 
    print('#'*60)
    print("The classification report of knn model: \n", classification_report_knn)


    model_precision = precision_score(y_test, pred_knn, average='macro')
    model_recall = recall_score(y_test, pred_knn, average='macro')
    model_f1 = f1_score(y_test, pred_knn, average='macro')

    model_precision1 = precision_score(y_test, dtree_pred, average='macro')
    model_recall1 = recall_score(y_test, dtree_pred, average='macro')
    model_f11 = f1_score(y_test, dtree_pred, average='macro')

    w_model_precision = precision_score(y_test, pred_knn, average='weighted')
    w_model_recall = recall_score(y_test, pred_knn, average='weighted')
    w_model_f1 = f1_score(y_test, pred_knn, average='weighted')


    w_model_precision1 = precision_score(y_test, dtree_pred, average='weighted')
    w_model_recall1 = recall_score(y_test, dtree_pred, average='weighted')
    w_model_f11 = f1_score(y_test, dtree_pred, average='weighted')

    data = {'KNN (macro)': [model_precision],
        'DTree (macro) ': [model_precision1],
        'KNN (weighted)': [w_model_precision],
        'DTree (weighted)': [w_model_precision1]}
    data2 = {'KNN (macro)': [model_recall],
        'DTree (macro) ': [model_recall1],
        'KNN (weighted)': [w_model_recall],
        'DTree (weighted)': [w_model_recall1]}
    data3= {'KNN (macro)': [model_f1],
        'DTree (macro) ': [model_f11],
        'KNN (weighted)': [w_model_f1],
        'DTree (weighted)': [w_model_f11]}
    data4 = {'KNN': [accuracy_knn],
        'DTree': [accuracy_dt]} 
    # Create the pandas DataFrame with column name is provided explicitly
    p_df = pd.DataFrame(data)
    recall_df = pd.DataFrame(data2)
    f1_df = pd.DataFrame(data3)
    acc_df = pd.DataFrame(data4)

    print(p_df.head())
    # print dataframe.

    ax = p_df.plot.bar()
    ax.set_title('Precision Score Comparsion',  fontsize=16)
    plt.legend()

    ax = recall_df.plot.bar()
    ax.set_title('Recall Score Comparsion',  fontsize=16)
    plt.legend()

    ax = f1_df.plot.bar()
    ax.set_title('F1 Score Comparsion',  fontsize=16)
    plt.legend()

    ax = acc_df.plot.bar()
    ax.set_title('Accuracy  Comparsion',  fontsize=16)
    plt.legend()

    plt.show()
    return