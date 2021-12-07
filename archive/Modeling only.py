#Loading Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
import seaborn as sns
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, accuracy_score, classification_report, coverage_error, label_ranking_average_precision_score, label_ranking_loss, dcg_score, ndcg_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import is_multilabel
from sklearn.multiclass import OneVsRestClassifier
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


dataset = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', header = 0)

#Splitting the Data
X = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(1,188), header = 0)
Y = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(188,198), header = 0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .3, random_state = 0)

print(X.shape, Y.shape)
is_multilabel(Y)

Y_labels = ['death.within.28.days','re.admission.within.28.days','death.within.3.months','re.admission.within.3.months','death.within.6.months','re.admission.within.6.months','return.to.emergency.department.within.6.months','outcome.during.hospitilization.alive','outcome.during.hospitilization.dead','outcome.during.hospitilization.DischargeAgainstOrder']



###MultiLabel Classification###


###Evaluation Functions##

def example_based_accuracy(y_test, pred):
    # compute true positives using the logical AND operator
    numerator = np.sum(np.logical_and(y_test, pred), axis = 1)
    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_test, pred), axis = 1)
    instance_accuracy = numerator/denominator
    avg_accuracy = np.mean(instance_accuracy)
    return avg_accuracy


def example_based_precision(y_test, pred):
    """
    precision = TP/ (TP + FP)
    """
    # Compute True Positive 
    precision_num = np.sum(np.logical_and(y_test, pred), axis = 1)
    # Total number of pred true labels
    precision_den = np.sum(pred, axis = 1)
    # precision averaged over all training examples
    avg_precision = np.mean(precision_num/precision_den)
    return avg_precision


def label_based_macro_precision(y_test, pred):
	# axis = 0 computes true positive along columns i.e labels
	l_prec_num = np.sum(np.logical_and(y_test, pred), axis = 0)
	# axis = computes true_positive + false positive along columns i.e labels
	l_prec_den = np.sum(pred, axis = 0)
	# compute precision per class/label
	l_prec_per_class = l_prec_num/l_prec_den
	# macro precision = average of precsion across labels. 
	l_prec = np.mean(l_prec_per_class)
	return l_prec


def label_based_micro_precision(y_test, pred):
    # compute sum of true positives (tp) across training examples and labels. 
    l_prec_num = np.sum(np.logical_and(y_test, pred))
    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(pred)
    # compute micro-averaged precision
    return l_prec_num/l_prec_den


def label_based_macro_recall(y_test, pred):
    # compute true positive along axis = 0 i.e labels
    l_recall_num = np.sum(np.logical_and(y_test, pred), axis = 0)
    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = np.sum(y_test, axis = 0)
    # compute recall per class/label
    l_recall_per_class = l_recall_num/l_recall_den
    # compute macro averaged recall i.e recall averaged across labels. 
    l_recall = np.mean(l_recall_per_class)
    return l_recall


def label_based_micro_recall(y_test, pred):
    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_test, pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_test)
    # compute mirco-average recall
    return l_recall_num/l_recall_den


def alpha_evaluation_score(y_test, pred):
    alpha = 1
    beta = 0.25
    gamma = 1
    # compute true positives across training examples and labels
    tp = np.sum(np.logical_and(y_test, pred))
    # compute false negatives (Missed Labels) across training examples and labels
    fn = np.sum(np.logical_and(y_test, np.logical_not(pred)))
    # compute False Positive across training examples and labels.
    fp = np.sum(np.logical_and(np.logical_not(y_test), pred))
    # Compute alpha evaluation score
    alpha_score = (1 - ((beta * fn + gamma * fp ) / (tp +fn + fp + 0.00001)))**alpha 
    return alpha_score


def label_based_macro_accuracy(y_test, pred):
    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = np.sum(np.logical_and(y_test, pred), axis = 0)
    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = np.sum(np.logical_or(y_test, pred), axis = 0)
    # compute mean accuracy across labels. 
    return np.mean(l_acc_num/l_acc_den)


def label_based_micro_accuracy(y_test, pred):
    # sum of all true positives across all examples and labels 
    l_acc_num = np.sum(np.logical_and(y_test, pred))
    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = np.sum(np.logical_or(y_test, pred))
    # compute mirco averaged accuracy
    return l_acc_num/l_acc_den


def exact_match_ratio(y_test, pred):
    n = len(y_test)
    row_indicators = np.all(y_test == pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count/n

def hamming_score(y_test, pred):
    return ((y_test & pred).sum(axis=1) / (y_test | pred).sum(axis=1)).mean()


def evaluation(y_test, pred):
    alpha_score = alpha_evaluation_score(y_test, pred)
    lb_micro_accuracy = label_based_micro_accuracy(y_test, pred)
    lb_macro_accuracy = label_based_macro_accuracy(y_test, pred)
    lb_micro_recall = label_based_micro_recall(y_test, pred)
    lb_macro_recall = label_based_macro_recall(y_test, pred)
    lb_micro_precision = label_based_micro_precision(y_test, pred)
    lb_macro_precision = label_based_macro_precision(y_test, pred)
    ex_precision = example_based_precision(y_test, pred)
    ex_accuracy = example_based_accuracy(y_test, pred)
    emr = exact_match_ratio(y_test, pred)
    coverage = coverage_error(y_test, pred)
    LRAP = label_ranking_average_precision_score(y_test, pred)
    hamming = hamming_loss(y_test, pred)
    label_loss = label_ranking_loss(y_test, pred)
    dcg = dcg_score(y_test, pred)
    ndcg = ndcg_score(y_test, pred)
    report = classification_report(y_test, pred)
    hammingsc = hamming_score(y_test, pred)

    print('label based alpha evaluation score', alpha_score)
    print('label based micro accuracy', lb_micro_accuracy)
    print('label based macro accuracy', lb_macro_accuracy)
    print('label based micro recall', lb_micro_recall)
    print('label based macro recall', lb_macro_recall)
    print('label based micro precision', lb_micro_precision)
    print('label based macro precision', lb_macro_precision)
    print('example based precision', ex_precision)
    print('example based accuracy', ex_accuracy)
    print('example based exact match ratio', emr)
    print('coverage error', coverage)
    print('LRAP', LRAP)
    print('hamming loss', hamming)
    print('label loss', label_loss)
    print('discounted cumulative gain', dcg)
    print('normalized discounted cumulative gain', ndcg)
    print(report)
    print('hamming score', hammingsc)


def hamming_score1(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

##SKLearn supported models with MultiOutputClassifier##


#Decision Tree
DT = DecisionTreeClassifier(random_state = 0)
DT_model = MultiOutputClassifier(DT, n_jobs=1)
DT_model = DT_model.fit(X_train, y_train)
DT_pred = DT_model.predict(X_test)
evaluation(y_test, DT_pred)
hamming_score1(y_test, DT_pred)


#Extra Tree
ET = ExtraTreeClassifier(random_state=0)
ET_model = BaggingClassifier(ET, random_state=0)
ET_model = MultiOutputClassifier(ET_model, n_jobs =1)
ET_model = ET_model.fit(X_train, y_train)
ET_pred = ET_model.predict(X_test)
evaluation(y_test, ET_pred)


#Extra Trees
ETS = ExtraTreesClassifier(random_state=0)
ETS_model = MultiOutputClassifier(ETS, n_jobs=1)
ETS_model = ETS_model.fit(X_train, y_train)
ETS_pred = ETS_model.predict(X_test)
evaluation(y_test, ETS_pred)


#KNN
KNN = KNeighborsClassifier(n_neighbors=3)
KNN_model = MultiOutputClassifier(KNN, n_jobs=1)
KNN_model = KNN_model.fit(X_train, y_train)
KNN_pred = KNN_model.predict(X_test)
evaluation(y_test, KNN_pred)


#Neural Network MLP
MLP = MLPClassifier(random_state = 1, max_iter = 300)
MLP_model = MultiOutputClassifier(MLP, n_jobs=1)
MLP_model = MLP_model.fit(X_train, y_train)
MLP_pred = MLP_model.predict(X_test)
evaluation(y_test, MLP_pred)


#SVC
SVC = SVC()
SVC_model = MultiOutputClassifier(SVC, n_jobs=1)
SVC_model = SVC_model.fit(X_train, y_train)
SVC_pred = SVC_model.predict(X_test)
evaluation(y_test, SVC_pred)


#RandomForest
RF = RandomForestClassifier(max_depth=2, random_state=0)
RF_model = MultiOutputClassifier(RF, n_jobs=1)
RF_model = RF_model.fit(X_train, y_train)
RF_pred = RF_model.predict(X_test)
evaluation(y_test, RF_pred)


#Ridge
RC = RidgeClassifierCV()
RC_model = MultiOutputClassifier(RC, n_jobs=1)
RC_model = RC_model.fit(X_train, y_train)
RC_pred = RC_model.predict(X_test)
evaluation(y_test, RC_pred)


#RadiusNeighbors
RN = RadiusNeighborsClassifier(radius=1.0)
RN_model = MultiOutputClassifier(RN, n_jobs=1)
RN_model = RN_model.fit(X_train, y_train)
RN_pred = RN_model.predict(X_test)
evaluation(y_test, RN_pred)












##SKLearn supported models with OneVsRestClassifier##

#Decision Trees
clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(X_train, y_train)
ODT_pred = clf.predict(X_test)
evaluation(y_test, ODT_pred)

# Extra Tree
clf = OneVsRestClassifier(ExtraTreeClassifier()).fit(X_train, y_train)
OET_pred = clf.predict(X_test)
evaluation(y_test, OET_pred)


#KNN
clf = OneVsRestClassifier(KNeighborsClassifier()).fit(X_train, y_train)
OKNN_pred = clf.predict(X_test)
evaluation(y_test, OKNN_pred)


#MLP
clf = OneVsRestClassifier(MLPClassifier()).fit(X_train, y_train)
OMLP_pred = clf.predict(X_test)
evaluation(y_test, OMLP_pred)


#Random Forrest
clf = OneVsRestClassifier(RandomForestClassifier()).fit(X_train, y_train)
ORF_pred = clf.predict(X_test)
evaluation(y_test, ORF_pred)


#SVC
clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
OSVC_pred = clf.predict(X_test)
evaluation(y_test, OSVC_pred)


#Radius
clf = OneVsRestClassifier(RadiusNeighborsClassifier()).fit(X_train, y_train)
ORN_pred = clf.predict(X_test)
evaluation(y_test, ORN_pred)


#Ridge
clf = OneVsRestClassifier(RidgeClassifierCV()).fit(X_train, y_train)
ORC_pred = clf.predict(X_test)
evaluation(y_test, ORC_pred)


#Extra Trees
clf = OneVsRestClassifier(ExtraTreesClassifier()).fit(X_train, y_train)
OETS_pred = clf.predict(X_test)
evaluation(y_test, OETS_pred)






#Predicting Labels

categories = list(Y.columns.values)

for category in categories:
    print(category)
    
    # Training logistic regression model on train data
    clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(X_train, y_train)
    
    # calculating test accuracy
    prediction = clf.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(y_test[category], prediction)))
    print("\n")





z = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = (44,45,46,70), header = 0)

zy = z.append(Y)

###Visualization###

#Pearson Correlation Matrix
correlations = Y.corr()
sns.heatmap(correlations, annot = True)
plt.title("Correlation matrix of Outcomes")
plt.show()


correlations = zy.corr()
sns.heatmap(correlations, annot = True)
plt.title("Correlation matrix")
plt.show()




Y.hist()
plt.show()

Y.plot(kind='density', subplots=True, layout=(3,4), sharex=False)
plt.show()


scatter_matrix(Y)
plt.show()

# plotting feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,15))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


#Feature Importance
for estimator in RF_model.estimators_:
    weights = pd.DataFrame(estimator.coef_, X_train.columns, columns=['Coefficients'])


len(RF_model.estimators_)
#10

X = [0,1,2,3,4,5,6,7,8,9]

RF_model.estimators_[0].feature_importances_


print(RF_model.feature_importances_)


svc = MultiOutputClassifier(SVC(kernel = 'linear'))
min_features_to_select = 10

rfecv = RFECV(estimator = svc, step =1, cv=StratifiedKFold(2), scoring = 'accuracy', min_features_to_select = min_features_to_select)
rfecv(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)



svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications

min_features_to_select = 10  # Minimum number of features to consider
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)

rfecv.fit(X, Y)



fig = plt.figure(figsize=(20,20))
(X_test, ax_train) = fig.subplots(ncols=2, nrows=1)
g1 = sns.barplot(x=Y.sum(axis=0), y=multilabel_binarizer.classes_, ax=ax_test)
g2 = sns.barplot(x=y_train_tfidf.sum(axis=0), y=multilabel_binarizer.classes_, ax=ax_train)
g1.set_title("class distribution before resampling")
g2.set_title("class distribution in training set after resampling")




import shap





clf = OneVsRestClassifier(RandomForestClassifier())
clf.fit(X_train, y_train)


for i in range(0, clf.coef_.shape[0]):
    top20_indices = np.argsort(clf.coef_[i])[-20:]
    print(top20_indices)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
feature_importances = pd.DataFrame(clf.feature_importances_,
                               index = X_train.columns,
                      columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)
clf.feature_importances_
sorted_idx = clf.feature_importances_.argsort()

plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show()


from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(clf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")



feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.xlabel("Feature Importance")
plt.show()


