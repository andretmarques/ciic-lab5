import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, auc
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-Sales-Win-Loss.csv")

le = LabelEncoder()
sc = StandardScaler()

df['Supplies Subgroup'] = le.fit_transform(df['Supplies Subgroup'])
df['Supplies Group'] = le.fit_transform(df['Supplies Group'])
df['Region'] = le.fit_transform(df['Region'])
df['Route To Market'] = le.fit_transform(df['Route To Market'])
df['Opportunity Result'] = le.fit_transform(df['Opportunity Result'])
df['Competitor Type'] = le.fit_transform(df['Competitor Type'])

df[['Opportunity Number', 'Elapsed Days In Sales Stage', 'Sales Stage Change Count',
    'Total Days Identified Through Closing', 'Total Days Identified Through Qualified', 'Opportunity Amount USD',
    'Client Size By Revenue', 'Client Size By Employee Count', 'Revenue From Client Past Two Years',
    'Deal Size Category']] = \
    sc.fit_transform(df[['Opportunity Number', 'Elapsed Days In Sales Stage', 'Sales Stage Change Count',
                         'Total Days Identified Through Closing', 'Total Days Identified Through Qualified',
                         'Opportunity Amount USD', 'Client Size By Revenue', 'Client Size By Employee Count',
                         'Revenue From Client Past Two Years', 'Deal Size Category']])

cols = [col for col in df.columns if col not in ['Opportunity Number', 'Opportunity Result']]
df_no_outliers = df[(np.abs(stats.zscore(df[cols])) < 3).all(axis=1)]
target = df_no_outliers['Opportunity Result']

df_no_outliers = df_no_outliers[cols]
data_train, data_test, target_train, target_test = train_test_split(df_no_outliers, target, test_size=0.3,
                                                                    random_state=10)

clf = MLPClassifier(random_state=1, max_iter=150).fit(data_train, target_train)
pred = clf.predict(data_test)

classification = classification_report(target_test, pred, output_dict=True)
precision = 'Precision : ' + str(classification['weighted avg']['precision'])
recall = 'Recall    : ' + str(classification['weighted avg']['recall'])
fscore = 'F-score   : ' + str(classification['weighted avg']['f1-score'])
accuracy = 'Accuracy  : ' + str(classification['accuracy'])

# parameter_space = {
#     'hidden_layer_sizes': [(50, 50, 50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant', 'adaptive'],
# }
#
# grid = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=5)
# grid.fit(data_train, target_train)
#
# print('Best parameters found:\n', grid.best_params_)
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


prob_pos = clf.predict_proba(data_test)[:, 1]

lr_precision, lr_recall, _ = precision_recall_curve(target_test, prob_pos)
lr_f1, lr_auc = f1_score(target_test, pred), auc(lr_recall, lr_precision)

# plot the precision-recall curves
plt.plot(lr_recall, lr_precision, label='Precision - Recall Curve')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

if __name__ == '__main__':
    print('Results on the test set:')
    print(precision)
    print(recall)
    print(fscore)
    print(accuracy)
