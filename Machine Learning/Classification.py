import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
# from Preprocessing import *
import math
from Features_Selection import prior_probability

#######################################################################################
#######################################################################################
############################  Bayesian Classifier  ####################################
#######################################################################################
#######################################################################################
def multivariate_normal_gaussian(X, mu, sigma):
    """
    Compute the conditional probabilty assuming normal distribution.

    Parameters
    ----------
    X : List
        the point at which we calculate the probabilty.
    mu : 2D List
        The mean of the features.
    sigma : float 
        standard deviation of the features.
    
    Returns
    -------
    prob : float
        The conditional probabilty.
    """
    prob = (1/(math.pow((2*math.pi),2/2)*math.pow(np.linalg.det(sigma),0.5)))*math.exp(-0.5*np.transpose(X-mu) @ np.linalg.inv(sigma) @(X-mu))
    return prob

# Compute P(C),mean and cov for each class.
def getPreProbMeanCov(allFeatures,classes,y_train,x_train):
    """
    Compute the prior probabilty, mean and covariance.

    Parameters
    ----------
    allFeatures : 2D List
        2D list containing all features for classes.
    classes : List
        List of all possible labels.
    x_train : List
        List of all training points.
    y_train : List
        List of all training labels.
    
    Returns
    -------
    PreProb : List
        The prior probabilty for each class.
    mean : List
        mean values for each feature.
    cov : 2D List
        covariances values between all features.
    """
    PreProb = []
    mean = []
    cov = []
    for c in range(len(classes)):
        PreProb.append(prior_probability(classes[c],y_train))
        temp = []
        for i in range(x_train.shape[1]):
            temp.append(np.mean(allFeatures[c][:,i]))
        mean.append(tuple(temp))
        cov.append(np.cov(np.transpose(allFeatures[c])))
    mean = np.array(mean)
    cov = np.array(cov)

    return PreProb,mean,cov

def BayesPredections(x_test,mean,cov,PreProb):
    """
    Apply NBPW classifier to predict classes for each point in x_test.

    Parameters
    ----------
    x_test : List
        List of all test points.
    PreProb : List
        The prior probabilty for each class.
    mean : List
        mean values for each feature.
    cov : 2D List
        covariances values between all features.

    Returns
    -------
    predicted_classes : List
        The classifer predictions for each test point.
    """
    predicted_classes = []
    for i in range(x_test.shape[0]):
        classProbabilities = np.zeros(2)
        # Compute the probability that the test point X_Test[i] belongs to each class in classes.
        classProbabilities[0]=multivariate_normal_gaussian(x_test[i],mean[0],cov[0])
        classProbabilities[1]=multivariate_normal_gaussian(x_test[i],mean[1],cov[1])

        total_prob = 0
        for i in range(2):
            total_prob = total_prob + (classProbabilities[i] * PreProb[i])
        for i in range(2):
            classProbabilities[i] = (classProbabilities[i] * PreProb[i])/total_prob
        
        # Classify the test point to the class has maximum probability.
        predicted_classes.append(int(classProbabilities.argmax())+1)
    return predicted_classes

# Naiive Bayes Parzen Window Classifier.
def NBPW (x_train,y_train,x_test,y_test,classes):
    """
    Apply NBPW classifier algorithm.

    Parameters
    ----------
    x_train : List
        List of all training points.
    y_train : List
        List of all training labels.
    x_test : List
        List of all test points.
    y_test : List
        List of all test labels.
    classes : List
        List of all possible labels.

    Returns
    -------
    accracy : float
        The classifer accuracy.
    """
    # Split features of each class.
    feature1 = x_train[y_train==classes[0]]
    feature2 = x_train[y_train==classes[1]]
    allFeatures = [feature1, feature2]

    # Compute P(C),mean and cov for each class.
    PreProb,mean,cov = getPreProbMeanCov(allFeatures,classes,y_train,x_train)
    
    # Classify
    predicted_classes = BayesPredections(x_test,mean,cov,PreProb)

    # Compute Accuracy
    predicted_classes = np.asarray(predicted_classes)
    y_test=np.asarray(y_test)
    accuracy = sum(y_test == predicted_classes)/len(y_test)

    return accuracy

#######################################################################################
#######################################################################################
###################################  LDA  #############################################
#######################################################################################
#######################################################################################

def LDAClassify(data_train,data_test,labels_train,labels_test):
    """
    Apply LDA classifier algorithm.

    Parameters
    ----------
    x_train : List
        List of all training points.
    y_train : List
        List of all training labels.
    x_test : List
        List of all test points.
    y_test : List
        List of all test labels.

    Returns
    -------
    accracy : float
        The classifer accuracy.
    """
    clf = LinearDiscriminantAnalysis()
    clf.fit(data_train, labels_train)
    accuracy = clf.score(data_test, labels_test)
    return accuracy

#######################################################################################
#######################################################################################
###################################  SVM  #############################################
#######################################################################################
#######################################################################################

def SVMClassify(data_train,data_test,labels_train,labels_test):
    """
    Apply SVM classifier algorithm.

    Parameters
    ----------
    x_train : List
        List of all training points.
    y_train : List
        List of all training labels.
    x_test : List
        List of all test points.
    y_test : List
        List of all test labels.

    Returns
    -------
    accracy : float
        The classifer accuracy.
    """
    clf = SVC()
    clf.fit(data_train, labels_train)
    accuracy = clf.score(data_test, labels_test)
    return accuracy