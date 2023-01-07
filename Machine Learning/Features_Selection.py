"""
This file implements the feature selection module by applying the Mutual Information-based Best Individual Feature (MIBIF)
to select the discriminative from the extracted features by the CSP algorithm.
"""
import math
import numpy as np

#Compute Prior Probability P(x)
def prior_probability(x,arr):
    """
    Compute the prior probabilty of x in a list P(x).

    Parameters
    ----------
    x : Any
        The item which we need to compute the probabilty for.
    arr : List or numby array
        The list which we search for x inside.

    Returns
    -------
    p : float
        The prior probabilty of X in arr.
    """
    p = list(arr).count(x)/len(arr)
    return p

# Calculate H and bumb function to make density estimation to compute P(x|C). 
def calculate_h_optimal(N, M, sigma):
    """
    Calculate the optimal value for the bandwidth (h) of the bump function used.

    Parameters
    ----------
    N : int
        Number of fetures.
    M : int
        Number of training points.
    sigma : float
        Standard deviation of all features that belongs to specific class.

    Returns
    -------
    h : float
        The optimal value to use in Parzen window.
    """
    
    h = (sigma) * ((4/(M*(N+2))) ** (1/(N+4)))
    return h

def bump_function(point, x, h):
    """
    Implement the bump function phi(x) at point p with bandwidth h to be used in density estimation. 

    Parameters
    ----------
    point : Any
        The point at which we compute the bump function (the mean).
    x : Any
        the variable for which we want to calclate phi
    h : float
        The optimal value to use in Parzen window.

    Returns
    -------
    phi : float
        The optimal value to use in Parzen window.

    Notes
    -------
    -x can be a scalar or a vector.
    -We choose phi to be a gaussian function assuming that the features have gaussian distribution.
    """
    phi = 1/(h * np.sqrt(2 * np.pi)) * np.exp( - (x - point)**2 / (2 * h**2))
    return phi

#Compute postrior Probability P(x|C) using density estimation.
def postrior_probability(l,labels,features,f,i):
    """
    Compute the postrior probability for a feature given the class P(x|C) using density estimation.

    Parameters
    ----------
    l : int
        The class we compute the probabilty given it.
    labels : List
        List of all training labels.
    features : 2D List
        list of all readings for all trials.
    f : float 
        The feature we compute the probabilty for it. (column index)
    i :
        The trial for which we compute the probabilty now. (row index)
    Returns
    -------
    p : float
        The postrior probabilty of x given Class l.
    """
    h = calculate_h_optimal(1, len(labels), np.sqrt(np.var(features[labels == l])))
    n = list(labels).count(l)
    targetTrials = features[labels == l]
    prob = 0
    for t in targetTrials:
        prob += bump_function(t[i],f,h)
    p = prob/n
    return p



#Compute Entropy for one class.
def Labels_Entropy(labels):
    """
    Compute the Entroby for all classes H(C) using.

    Parameters
    ----------
    labels : List
        List of all training labels.

    Returns
    -------
    entroby : float
        The entroby for all classes.
    """
    entropy =0
    for label in labels:
        prob = prior_probability(label,labels)
        entropy -= prob*math.log2(prob)
    return entropy

#Compute Conditional Entropy for one feature.
def Conditional_Entropy(features, labels, f, classes):
    """
    Compute the conditional entroby for a class H(C|x) given a feature.

    Parameters
    ----------
    labels : List
        List of all training labels.
    features : 2D List
        list of all readings for all trials.
    f : float 
        The feature we compute the entroby given it. (column index)
    classes : list 
        A list of the classes that we'll classify between
    Returns
    -------
    entroby : float
        The conditional entroby of C given feature f.
    """
    entropy = 0
    for label in classes:
        for i in range(features.shape[0]):
            # temp is a variable to hold the value of P(f) to be used as dominator later.
            temp=0
            for l in classes:
                temp += postrior_probability(l,labels,features, features[i][f],f)*prior_probability(l,labels)
            # P(C|X) = P(X|C)* P(C) / P(f)
            prob = (postrior_probability(label,labels,features,features[i][f],f)*prior_probability(label,labels))/temp   
            entropy -= prob*math.log2(prob)
    return entropy


def mutual_information(features, labels, classes, k=4):
    """
    Compute the mutual information between features and select best K features.

    Parameters
    ----------
    labels : List
        List of all training labels.
    features : 2D List
        list of all readings for all training trials.
    test : 2D List
        list of all readings for all test trials.
    k : int
        Number of features to be selected.
    classes : list 
        A list of the classes that we'll classify between
    Returns
    -------
    X_train : 2D List
        List of new training data.
    X_test : 2D List
        List of new test data.
    """
    I=[]
    for i in range(features.shape[1]):
        I.append(Labels_Entropy(labels)- Conditional_Entropy(features, labels, i, classes))
    indexes = list(range(len(I)))
    indexes.sort(key=I.__getitem__, reverse=True)
    X_train = features[:, indexes[:k]]

    # Return top k indexes to be used on test
    return X_train, indexes[:k]
