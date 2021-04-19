import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, auc
from yellowbrick.classifier import ClassificationReport
from scipy import stats
from sklearn.model_selection import validation_curve
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

clf = MLPClassifier(hidden_layer_sizes=50, activation='tanh', solver='adam', learning_rate='constant', alpha=0.05,
                    random_state=1, max_iter=150, early_stopping=True, validation_fraction=0.3) \
    .fit(data_train, target_train)
pred = clf.predict(data_test)

classification = classification_report(target_test, pred, output_dict=True)

precision = 'Precision : ' + str(classification['weighted avg']['precision'])
recall = 'Recall    : ' + str(classification['weighted avg']['recall'])
fscore = 'F-score   : ' + str(classification['weighted avg']['f1-score'])
accuracy = 'Accuracy  : ' + str(classification['accuracy'])

classification_ = ClassificationReport(clf, classes=['Won', 'Loss'])
classification_.fit(data_train, target_train)
classification_.score(data_test, target_test)
classification_.show()

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


def alphas():
    h = .02  # step size in the mesh

    alphas = np.logspace(-1, 1, 5) - 0.05
    classifiers = []
    names = []

    for alpha in alphas:
        classifiers.append(make_pipeline(
            StandardScaler(),
            MLPClassifier(
                solver='adam', alpha=alpha, random_state=1, max_iter=10000,
                early_stopping=True, hidden_layer_sizes=[50, 50]
            )
        ))
        names.append(f"alpha {alpha:.2f}")

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=0, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable]

    figure = plt.figure(figsize=(17, 9))
    i = 1
    # iterate over datasets
    colorSet = plt.cm.RdBu
    for X, y in datasets:
        # split into training and test part
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, classi in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            classi.fit(X_train, y_train)
            score = classi.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max] x [y_min, y_max].
            if hasattr(classi, "decision_function"):
                Z = classi.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = classi.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=colorSet, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='black', s=25)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6, edgecolors='black', s=25)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    plt.show()


def grid():
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    grid_ = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=5)
    grid_.fit(data_train, target_train)

    print('Best parameters found:\n', grid_.best_params_)
    means = grid_.cv_results_['mean_test_score']
    stds = grid_.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


if __name__ == '__main__':
    print('Results on the test set:')
    print(precision)
    print(recall)
    print(fscore)
    print(accuracy)
    alphas()  # source: shorturl.at/gtyHX
    # grid()  # Attention, this requires too much CPU power and draws lots of energy for a long time
