# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from math import log
from sklearn.metrics import accuracy_score

def load_data(filename):
    mat = np.loadtxt(filename,delimiter=',')
    data = shuffle(mat)
    Xs = data[:,:-1] #Actual data
    Ys = data[:,-1] #Classes
    means = np.mean(Xs,0)
    stds = np.std(Xs,0)
    #Standardize the data: z1 = (x1 - mean)/deviation
    Xs = (Xs - means) / stds
    return Xs, Ys

def calc_fold_log(feats,X,Y,train_ix,valid_ix,C=1e12):
    """return classification error for train and validation/test sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats], Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])

def Logistic(kf, X_r, Y_r, X_t, Y_t):
    C = 1
    best_C = C
    Cs = []
    train_errs = []
    valid_errs = []
    best_err = 10000
    for i in range(20):
        train_err = valid_err = 0
        for train_ix, valid_ix in kf:
            class_err, test_err = calc_fold_log(4, X_r, Y_r, train_ix, valid_ix, C)
            train_err += class_err
            valid_err += test_err
        train_err_avg = np.mean(train_err)
        valid_err_avg = np.mean(valid_err)
        if valid_err_avg < best_err:
            best_err = valid_err_avg
            best_C = C
        print(4,':',train_err_avg,valid_err_avg, ':', C)
        Cs.append(log(C))
        train_errs.append(train_err_avg)
        valid_errs.append(valid_err_avg)
        C *= 2
    print('Best C = ',best_C)
    plt.figure(figsize=(8,8))
    plt.title('Error vs C Value')
    plt.xlabel('Log(C) Value')
    plt.ylabel('Error value')
    plt.plot(Cs,train_errs,'bo-',label='Training Error',linewidth=1,markersize=4)
    plt.plot(Cs,valid_errs,'ro-',label='Validation Error',linewidth=1,markersize=4)
    plt.legend(loc='best')
    plt.figtext(0.5,0.03,
         'Best C: {0},  Best Validation Error: {1:8.4f}'
         .format(best_C, best_err),
         horizontalalignment='center')
    plt.savefig('Logistic Regression.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()
    reg = LogisticRegression(C=best_C, tol=1e-10)
    reg.fit(X_r,Y_r)
    return 1 - reg.score(X_t,Y_t), best_C, reg.predict(X_t)

def calc_fold_knn(feats,X,Y,train_ix,valid_ix,N):
    """return classification error for train and validation/test sets"""
    knn = KNeighborsClassifier(N)
    knn.fit(X[train_ix,:feats], Y[train_ix])
    prob = knn.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])

def KNN(kf, X_r, Y_r, X_t, Y_t):
    N = 1
    Ns = []
    best_err = 10000
    train_errs = []
    valid_errs = []
    for ix in range(20):
        knn = KNeighborsClassifier(N)
        scores = cross_validate(knn,X_r,Y_r,cv=kf,return_train_score=True)
        train_err = 1 - np.mean(scores['train_score'])
        va_err = 1 - np.mean(scores['test_score'])
        if va_err < best_err:
            best_err = va_err
            best_N = N
        print(4,':',train_err, va_err, ':', N)
        Ns.append(N)
        train_errs.append(train_err)
        valid_errs.append(va_err)
        N += 2
        knn.fit(X_r,Y_r)
    print('Best N = ',best_N)
    plt.figure(figsize=(8,8))
    plt.title('Error vs Number of Neighbours')
    plt.xlabel('Number of Neighbours')
    plt.ylabel('Error Value')
    plt.plot(Ns,train_errs,'bo-',label='Training Error',linewidth=1,markersize=4)
    plt.plot(Ns,valid_errs,'ro-',label='Validation Error',linewidth=1,markersize=4)
    plt.legend(loc='best')
    plt.figtext(0.5,0.03,
     'Best Number of Neighbours: {0},  Best Validation Error: {1:8.4f}'
     .format(best_N, best_err),
     horizontalalignment='center')
    plt.savefig('K-Nearest Neighbours.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()
    knn = KNeighborsClassifier(best_N)
    knn.fit(X_r,Y_r)
    return 1 - knn.score(X_t,Y_t), best_N, knn.predict(X_t)

def NaiveBayes_train(X_train, Y_train, bandwidth):
    class0_train = Y_train[Y_train==0]
    class1_train = Y_train[Y_train==1]

    "Calculate prior probability"
    class_prob_C0 = len(class0_train)/len(Y_train)
    class_prob_C1 = len(class1_train)/len(Y_train)

    X1C0 = X_train[Y_train==0,0].reshape(-1,1)
    X1C1 = X_train[Y_train==1,0].reshape(-1,1)
    X2C0 = X_train[Y_train==0,1].reshape(-1,1)
    X2C1 = X_train[Y_train==1,1].reshape(-1,1)
    X3C0 = X_train[Y_train==0,2].reshape(-1,1)
    X3C1 = X_train[Y_train==1,2].reshape(-1,1)
    X4C0 = X_train[Y_train==0,3].reshape(-1,1)
    X4C1 = X_train[Y_train==1,3].reshape(-1,1)
    
    "Calculate log probability distribution of features"
    KDE_X1_C0 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X1_C0.fit(X1C0)
    KDE_X1_C1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X1_C1.fit(X1C1)
    KDE_X2_C0 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X2_C0.fit(X2C0)
    KDE_X2_C1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X2_C1.fit(X2C1)
    KDE_X3_C0 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X3_C0.fit(X3C0)
    KDE_X3_C1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X3_C1.fit(X3C1)
    KDE_X4_C0 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X4_C0.fit(X4C0)
    KDE_X4_C1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    KDE_X4_C1.fit(X4C1)

    class_probs = [class_prob_C0, class_prob_C1]
    KDEs = [KDE_X1_C0, KDE_X1_C1, KDE_X2_C0, KDE_X2_C1, KDE_X3_C0, KDE_X3_C1, KDE_X4_C0, KDE_X4_C1]
    
    return class_probs, KDEs
    
def NaiveBayes_predict(class_probs, KDEs, X_test):
    prob_X1_C0 = KDEs[0].score_samples(X_test[:,0].reshape(-1,1))
    prob_X1_C1 = KDEs[1].score_samples(X_test[:,0].reshape(-1,1))
    prob_X2_C0 = KDEs[2].score_samples(X_test[:,1].reshape(-1,1))
    prob_X2_C1 = KDEs[3].score_samples(X_test[:,1].reshape(-1,1))
    prob_X3_C0 = KDEs[4].score_samples(X_test[:,2].reshape(-1,1))
    prob_X3_C1 = KDEs[5].score_samples(X_test[:,2].reshape(-1,1))
    prob_X4_C0 = KDEs[6].score_samples(X_test[:,3].reshape(-1,1))
    prob_X4_C1 = KDEs[7].score_samples(X_test[:,3].reshape(-1,1))
    
    pred_y = []
    for i in range(len(X_test)):
        pred_C0 = np.log(class_probs[0]) + prob_X1_C0[i] + prob_X2_C0[i] + prob_X3_C0[i] + prob_X4_C0[i]
        pred_C1 = np.log(class_probs[1]) + prob_X1_C1[i] + prob_X2_C1[i] + prob_X3_C1[i] + prob_X4_C1[i]
        if pred_C0 > pred_C1:
            pred_y.append(0)
        else:
            pred_y.append(1)
            
    return pred_y

def calc_fold_nb(X,Y, train_ix,valid_ix,bandwidth):
    """return error for train and validation sets"""
    class_probs, KDEs = NaiveBayes_train(X[train_ix], Y[train_ix], bandwidth)
    train_pred_Y = NaiveBayes_predict(class_probs, KDEs, X[train_ix])
    valid_pred_Y = NaiveBayes_predict(class_probs, KDEs, X[valid_ix])
    return 1 - accuracy_score(Y[train_ix], train_pred_Y), 1 - accuracy_score(Y[valid_ix], valid_pred_Y)

def NaiveBayes(kf, X_r, Y_r, X_t, Y_t):
    bandwidth=0.01
    best_bandwidth = bandwidth
    bandwidths = []
    train_errs = []
    valid_errs = []
    best_err = 10000
    while bandwidth <= 1.0:
        train_err = valid_err = 0
        for train_ix, valid_ix in kf:
            class_err, test_err = calc_fold_nb(X_r, Y_r, train_ix, valid_ix, bandwidth)
            train_err += class_err
            valid_err += test_err
        train_err_avg = np.mean(train_err)
        valid_err_avg = np.mean(valid_err)
        if valid_err_avg < best_err:
            best_err = valid_err_avg
            best_bandwidth = bandwidth
        print(4,':',train_err_avg,valid_err_avg, ':', bandwidth)
        bandwidths.append(bandwidth)
        train_errs.append(train_err_avg)
        valid_errs.append(valid_err_avg)
        bandwidth = round(bandwidth+0.02,2)
    print('Best bandwidth = ',best_bandwidth)
    plt.figure(figsize=(8,8))
    plt.title('Error vs Bandwidth')
    plt.xlabel('Bandwidth Value')
    plt.ylabel('Error Value')
    plt.plot(bandwidths,train_errs,'bo-',label='Training Error',linewidth=1,markersize=4)
    plt.plot(bandwidths,valid_errs,'ro-',label='Validation Error',linewidth=1,markersize=4)
    plt.legend(loc='best')
    plt.figtext(0.5,0.03,
        'Best Bandwidth: {0},  Best Validation Error: {1:8.4f}'
        .format(best_bandwidth, best_err),
        horizontalalignment='center')
    plt.savefig('Naïve Bayes.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()
    class_probs, KDEs = NaiveBayes_train(X_r,Y_r,best_bandwidth)
    test_pred = NaiveBayes_predict(class_probs, KDEs, X_t)
    return 1 - accuracy_score(Y_t,test_pred), best_bandwidth, test_pred
    
def McNemar(PredA, PredB, y):
    TrueA = PredA == y
    FalseB = PredB != y
    TrueB = PredB == y
    FalseA = PredA != y
    NTaFb = sum(TrueA*FalseB)
    NTbFa = sum(TrueB*FalseA)
    return ((abs(NTaFb-NTbFa)-1)**2)*1.0/(NTaFb+NTbFa)

def compare(filename):
    Xs, Ys = load_data(filename)
    X_r, X_t, Y_r, Y_t = train_test_split(Xs, Ys, test_size=.33, stratify=Ys)
    folds = 5
    kf = StratifiedKFold(Y_r, n_folds=folds)
    LogErr, bestC, LogPred = Logistic(kf, X_r, Y_r, X_t, Y_t)
    KnnErr, bestN, KnnPred = KNN(kf, X_r, Y_r, X_t, Y_t)
    NBErr, bestB, NBPred = NaiveBayes(kf, X_r, Y_r, X_t, Y_t)
    print('Log Error: {}, Best C: {}'.format(LogErr, bestC))
    print('KNN Error: {}, Best N: {}'.format(KnnErr, bestN))
    print('NB Error: {}, Best B: {}'.format(NBErr, bestB))
    MCNemarLogKnn = McNemar(LogPred, KnnPred, Y_t)
    print('Log vs KNN Scores: ',MCNemarLogKnn)
    MCNemarKnnNB = McNemar(KnnPred, NBPred, Y_t)
    print('KNN vs NB Scores: ',MCNemarKnnNB)
    MCNemarNBLog = McNemar(NBPred, LogPred, Y_t)
    print('NB vs Log Scores: ',MCNemarNBLog)
    
compare('TP1-data.csv')