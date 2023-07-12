import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from numpy import mean
from sklearn.model_selection import ShuffleSplit
import timeit
from sklearn.decomposition import PCA
import csv
print("package loading done")

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


x = []
x1 = np.load("/alina-data0/Sarwan/chaos_wdgrl/x_train_wdgrl.npy", allow_pickle=True)
x2 = np.load("/alina-data0/Sarwan/chaos_wdgrl/x_val_wdgrl.npy", allow_pickle=True)
x3 = np.load("/alina-data0/Sarwan/chaos_wdgrl/x_test_wdgrl.npy", allow_pickle=True)
for i in range(len(x1)):
    x.append(x1[i])
for i in range(len(x2)):
    x.append(x2[i])
for i in range(len(x3)):
    x.append(x3[i])
seq_data = np.array(x)


#seq_data = np.load("/alina-data2/taslim/TSNE_Alternative_Kernel/Host/my_hashing2vec_onehotVec_host_One_Hot_Fecture_Vectors.npy", allow_pickle=True)
print(len(seq_data))
print("seq_data shape after padding",seq_data.shape)

#length_checker = np.vectorize(len)
#arr_len = length_checker(seq_data)
#X_padded = []
#for i in range(len(seq_data)):
#    tmp = padarray(seq_data[i], max(arr_len))
#    X_padded.append(tmp)
#print("max len", max(arr_len))
#print("len of X_padded", len(X_padded))
#seq_data = np.array(X_padded)
#print("seq_data shape after padding", seq_data.shape)

########### load chaos pkdd sarscov2 dataset
#file = open('./Attributes_reduced_23331.csv')
#csvreader = csv.reader(file)
#rows = []
#for row in csvreader:
#        row = ("".join(row))
#        rows.append(row)
#print(len(rows))

attribute_data = np.load("./all_variantNames.npy", allow_pickle=True)
print(np.unique(attribute_data, return_counts=True))
print(attribute_data[0])

attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[", "")
    aa_1 = aa.replace("]", "")
    aa_2 = aa_1.replace("\'", "")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)

print(np.unique(attr_new, return_counts=True))
print(" ")
print(np.unique(np.array(int_hosts), return_counts=True))

print("Attribute data preprocessing Done")

def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)


def svm_fun_kernel(X_train, y_train, X_test, y_test, kernel_mat):
    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)
    #     clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_mat)

    # Train the model using the training sets
    clf.fit(kernel_mat, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    svm_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("SVM Accuracy:",svm_acc)

    svm_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("SVM Precision:",svm_prec)

    svm_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("SVM F1 Weighted:",svm_f1_weighted)

    svm_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("SVM F1 macro:",svm_f1_macro)

    svm_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("SVM F1 micro:",svm_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix SVM : \n", confuse)
    # print("SVM Kernel Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    #    print(macro_roc_auc_ovo[1])
    check = [svm_acc, svm_prec, svm_recall, svm_f1_weighted, svm_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  SVM Classifier  ################################
def svm_fun(X_train, y_train, X_test, y_test):
    # scaler = RobustScaler()
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    start = timeit.default_timer()

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start
    #     print("NB Time : ", stop - start)

    svm_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("SVM Accuracy:",svm_acc)

    svm_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("SVM Precision:",svm_prec)

    svm_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("SVM F1 Weighted:",svm_f1_weighted)

    svm_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("SVM F1 macro:",svm_f1_macro)

    svm_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("SVM F1 micro:",svm_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix SVM : \n", confuse)
    # print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    #    print(macro_roc_auc_ovo[1])
    check = [svm_acc, svm_prec, svm_recall, svm_f1_weighted, svm_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  NB Classifier  ################################
def gaus_nb_fun(X_train, y_train, X_test, y_test):
    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    NB_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("Gaussian NB Precision:",NB_prec)

    NB_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("Gaussian NB Recall:",NB_recall)

    NB_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("Gaussian NB F1 weighted:",NB_f1_weighted)

    NB_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("Gaussian NB F1 macro:",NB_f1_macro)

    NB_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("Gaussian NB F1 micro:",NB_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix NB : \n", confuse)
    # print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc, NB_prec, NB_recall, NB_f1_weighted, NB_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  MLP Classifier  ################################
def mlp_fun(X_train, y_train, X_test, y_test):
    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)

    # Feature scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test_2 = scaler.transform(X_test)

    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test_2)

    stop = timeit.default_timer()
    time_new = stop - start

    MLP_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("MLP Accuracy:",MLP_acc)

    MLP_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("MLP Precision:",MLP_prec)

    MLP_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("MLP Recall:",MLP_recall)

    MLP_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("MLP F1:",MLP_f1_weighted)

    MLP_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("MLP F1:",MLP_f1_macro)

    MLP_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("MLP F1:",MLP_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix MLP : \n", confuse)
    # print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [MLP_acc, MLP_prec, MLP_recall, MLP_f1_weighted, MLP_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  knn Classifier  ################################
def knn_fun(X_train, y_train, X_test, y_test):
    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    knn_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("Knn Accuracy:",knn_acc)

    knn_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("Knn Precision:",knn_prec)

    knn_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("Knn Recall:",knn_recall)

    knn_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("Knn F1 weighted:",knn_f1_weighted)

    knn_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("Knn F1 macro:",knn_f1_macro)

    knn_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("Knn F1 micro:",knn_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix KNN : \n", confuse)
    # print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [knn_acc, knn_prec, knn_recall, knn_f1_weighted, knn_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  Random Forest Classifier  ################################
def rf_fun(X_train, y_train, X_test, y_test):
    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    fr_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("Random Forest Accuracy:",fr_acc)

    fr_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("Random Forest Precision:",fr_prec)

    fr_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("Random Forest Recall:",fr_recall)

    fr_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("Random Forest F1 weighted:",fr_f1_weighted)

    fr_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("Random Forest F1 macro:",fr_f1_macro)

    fr_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("Random Forest F1 micro:",fr_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix RF : \n", confuse)
    # print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [fr_acc, fr_prec, fr_recall, fr_f1_weighted, fr_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train, y_train, X_test, y_test):
    # scaler = RobustScaler()
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    start = timeit.default_timer()

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    LR_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("Logistic Regression Accuracy:",LR_acc)

    LR_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression Precision:",LR_prec)

    LR_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression Recall:",LR_recall)

    LR_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression F1 weighted:",LR_f1_weighted)

    LR_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("Logistic Regression F1 macro:",LR_f1_macro)

    LR_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("Logistic Regression F1 micro:",LR_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix LR : \n", confuse)
    # print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [LR_acc, LR_prec, LR_recall, LR_f1_weighted, LR_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


def fun_decision_tree(X_train, y_train, X_test, y_test):
    from sklearn import tree

    start = timeit.default_timer()
    #     stop = timeit.default_timer()
    #     print("NB Time : ", stop - start)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    dt_acc = metrics.accuracy_score(y_test, y_pred)
    #     print("Logistic Regression Accuracy:",LR_acc)

    dt_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression Precision:",LR_prec)

    dt_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression Recall:",LR_recall)

    dt_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    #     print("Logistic Regression F1 weighted:",LR_f1_weighted)

    dt_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    #     print("Logistic Regression F1 macro:",LR_f1_macro)

    dt_f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #     print("Logistic Regression F1 micro:",LR_f1_micro)

    confuse = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix DT : \n", confuse)
    # print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [dt_acc, dt_prec, dt_recall, dt_f1_weighted, dt_f1_macro, macro_roc_auc_ovo[1], time_new]
    return (check)


X = seq_data # np.array(seq_data)
print("XXxxx")
print(X[0])
print(len(X))
print(X[0])
y = np.array(int_hosts)
print(len(y))

svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []
dt_table = []

total_splits = 1
from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=total_splits, test_size=0.3)
sss.get_n_splits(X, y)

for splits_ind in range(total_splits):
    train_index, test_index = next(sss.split(X, y))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("len xtrain", len(X_train))
    print("len xtest", len(X_test))

    start = timeit.default_timer()
    gauu_nb_return = gaus_nb_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("NB Time : ", stop - start)
    print(gauu_nb_return)

    start = timeit.default_timer()
    mlp_return = mlp_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("MLP Time : ", stop - start)
    print(mlp_return)

    start = timeit.default_timer()
    knn_return = knn_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("KNN Time : ", stop - start)
    print(knn_return)

    start = timeit.default_timer()
    rf_return = rf_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("RF Time : ", stop - start)
    print(rf_return)

    start = timeit.default_timer()
    lr_return = lr_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("LR Time : ", stop - start)
    print(lr_return)

    start = timeit.default_timer()
    dt_return = fun_decision_tree(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("DT Time : ", stop - start)
    print(dt_return)

    start = timeit.default_timer()
    svm_return = svm_fun(X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    print("SVM Time : ", stop - start)
    print(svm_return)    

    # gauu_nb_table.append(gauu_nb_return)
    # mlp_table.append(mlp_return)
    # knn_table.append(knn_return)
    # rf_table.append(rf_return)
    # lr_table.append(lr_return)
    # dt_table.append(dt_return)
    # svm_table.append(svm_return)

#     svm_table_final = DataFrame(svm_table, columns=["Accuracy", "Precision", "Recall",
#                                                     "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#     gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy", "Precision", "Recall",
#                                                             "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#     mlp_table_final = DataFrame(mlp_table, columns=["Accuracy", "Precision", "Recall",
#                                                     "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#     knn_table_final = DataFrame(knn_table, columns=["Accuracy", "Precision", "Recall",
#                                                     "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#     rf_table_final = DataFrame(rf_table, columns=["Accuracy", "Precision", "Recall",
#                                                   "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#     lr_table_final = DataFrame(lr_table, columns=["Accuracy", "Precision", "Recall",
#                                                   "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#
#     dt_table_final = DataFrame(dt_table, columns=["Accuracy", "Precision", "Recall",
#                                                   "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"])
#
# final_mean_mat = []
#
# final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
# final_mean_mat.append(np.transpose((list(dt_table_final.mean()))))
#
# final_avg_mat = DataFrame(final_mean_mat, columns=["Accuracy", "Precision", "Recall",
#                                                    "F1 (weighted)", "F1 (Macro)", "ROC AUC", "Runtime"],
#                           index=["SVM", "NB", "MLP", "KNN", "RF", "LR", "DT"])
# print(final_avg_mat)
