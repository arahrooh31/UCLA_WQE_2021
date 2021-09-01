import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

#Splitting the Data
X = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(1,188), header = 0)
Y = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(188,198), header = 0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .3, random_state = 0)


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))



plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")



plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()






import seaborn as sns


outcomes_total = pd.read_csv('Book2.csv')
sns.displot(outcomes_total)

sns.barplot(x = 'Outcomes', y = 'Instances',  data = outcomes_total)


plt.show()


x = ['death within 28 days', 'readmission within 28 days', 'death within 3 months', 'readmission within 3 months', 'death within 6 months', 'readmission within 6 months', 'return to emergency department within 6 months', 'alive during hospitilization', 'death during hospitilization', 'discharge against order during  hospitilization']
y = [37, 140, 42, 498, 57, 773, 775, 1890, 11, 107]
plt.bar(x,y)
plt.show()

df = pd.read_csv('heartfailuredata.csv')

df['gender'].value_counts()
#male 845
#female 1163



from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
X = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(1,188), header = 0)
Y = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(188,198), header = 0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .3, random_state = 0)
label_names = ['death.within.28.days','re.admission.within.28.days','death.within.3.months','re.admission.within.3.months','death.within.6.months','re.admission.within.6.months','return.to.emergency.department.within.6.months','outcome.during.hospitilization.alive','outcome.during.hospitilization.dead','outcome.during.hospitilization.DischargeAgainstOrder']

graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
edge_map = graph_builder.transform(y_train)
print("{} labels, {} edges".format(len(label_names), len(edge_map)))
print(edge_map)
