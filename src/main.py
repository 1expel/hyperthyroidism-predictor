from data.make_dataset import main
from models.train_model import train
from models.predict_model import predict
from visualization.visualize import visualize

main()

fit_dt, fit_knn = train()

y_test, dtree_pred, pred_knn = predict(fit_dt, fit_knn)

visualize(y_test, dtree_pred, pred_knn)

print("\n----- CONCLUSIONS BASED ON OUR MODEL PERFORMANCE -----\n")

print("""
Since our data was multi-class classification we chose to use K-Nearest Neighbours and Decision Tree Models.
Patients could be classified into 1 of 5 classes - 0: T3 Toxic, 1: Goitre, 2: Hyperthyroid, 3: Negative, 4: Secondary Toxic.
Unfortunately, due to the data provided to us at https://archive.ics.uci.edu/ml/datasets/thyroid+disease the train data had no cases of Secondary Toxic and the test data had 1 case.
Due to this, our model was never trained on instances of secondary toxic cases and therefore, never would predict a secondary toxic case.
This is ok though, as it was the least common classification out of all 5 classes, and only appeared once in train and test (combined).
Onto details of our models...
We trained both K-Nearest Neighbours and Decision Tree Models using Hyperparam tuning to find the ideal parameters for each model, in order to get the best performance metrics.
We found that k=3 neighbours was optimal and decision tree depth = 4 was optimal.
In comparing the models, both models were similar in terms of overall accuracy, but Decision Tree was substantially superior in terms of metrics such as precission, recall, and f1 for INDIVIUAL class predictions.
Overall accuracy was heavily influnced by accuracy of Negative predictions as negative case were 947 out of 972 patients in y_test.
Both models had extremely high accuracy due to their very good ability to predict negative cases (knn = 97.5 percent accuracy and decision tree = 98.6 percent accuracy)
Where Decision Tree Model was substantially bettwer than KNN was for Goitre and Hyperthyroid class predictions.
These 2 classes were the 2nd and 3rd most occuring class behind Negative at 17/972 instances in y_test and 5/972 instances in y_test (respectively).
So the face Decision Tree had high scores for such low instances of these classes was impressive in our minds. (also since it was trained on low instances as well)
The Decision Tree has a precision score of 100 percent for the 5 cases of Goitre, meaning it had no false positives, and whenever it predicted positive, it was true!
The Decision Tree has a precision score of 80% percent for the 17 cases of Hyperthyroid, which is very solid for a low amount of instances.
Once again, overall, both models had extremly high accuracy scores of 97.5 percent+, meaning they overall were both great and had a f1 score of 99 percent for Negative cases
which means that patients are positevly predicted negative very accurately and the coverage of all negative cases is high.
But Decision Tree was overall better because when patients actually had a problem such as Goitre or Hyperthyroid, the Decision Tree model was better at predicting so.
This is why we decided to show the macro avg, since Negative cases are heavily weighted in the Weighted Average since they are 947 out of 972 patients in y_test,
we used Macro Avg to show how well the models predicted the other classes such as Goitre, Hyperthyroid, Secondary Toxic, and T3 Toxic.
The Macro avg was better in Decision Tree since it was better at predicted Goitre and Hyperthyroid.
But neither models could predict the 1 case of Secondary Toxic as it was never present in the train data and therefore, impossible to predict (which is ok, not a common health problem, only 1 case).
And neither models predicted the 2 cases of T3 Toxic, which is ok, since there were only 2 cases and this would be extremely hard.
Goitre at 5 occurences, and Hyperthyroid at 17 occurences were the most occuring and therefore, more important to predict, and Decision Tree did this well, shown in the F1 score for these classes in the graphs.
""")
