"""
This file implements the feature extraction module by applying the Common Spatial pattern (CSP) Z =W.X
to compute the transformation matrix W to find features whose variances are optimal 
for discriminating our classes of EEG measurements by solving the eigenvalue decomposition problem.
"""
import pywt
import numpy as np
from scipy.linalg import eig
from sklearn.model_selection import train_test_split


def ComputeZ(W, E):
    """
    Multiply data and CSP projection to get new features.

    Parameters
    ----------
    E : 2D List
        The data to be transformed.
    W : 2D List
        CSP projection matrix.

    Returns
    -------
    Z : 2D List
        data after tranformation to CSP space.
    """
    Z = []

    for i in range(E.shape[0]):
        Z.append(W @ E[i])
    return np.array(Z)

def getFeatureVector(Z):
    """
    Calculate the feature vector of Z.

    Parameters
    ----------
    Z : 2D List
        The data to be transformed.

    Returns
    -------
    feature : 2D List
        The final transformed features
    """
    feature = []
    
    for i in range(Z.shape[0]):
        var = np.var(Z[i], axis=1)
        varsum = np.sum(var)
        
        feature.append(np.log10(var/varsum))
        
    return np.array(feature)

def getCovarianceMatrix(data):
    """
    Compute covariance matrix for one class.

    Parameters
    ----------
    data: 2D List
        The training data.

    Returns
    -------
    cov : 2D List (features,features)
        Covariance matrix between the features.
    """
    cov = []
    for i in range(data.shape[0]):
        cov.append(data[i]@data[i].T/np.trace(data[i]@data[i].T))
    cov = np.mean(np.array(cov), 0)
    return cov
    
def decomposeMatrix(covs):
    """
    Compute eigen values and egin vectors for the covariance matrices.

    Parameters
    ----------
    covs : 3D List
        The list of all covariance matrices for all classes.

    Returns
    -------
    eigenValues : List 
        All eigen values for the covariance matrix.
    eigenVectors : 2D List 
        All eigen vectors for the covariance matrix.
    """
    eigenValues,eigenVectors = eig(covs[0], covs.sum(0))
    return eigenValues,eigenVectors

def getAllCovarianceMatrices(X, y, classes):
    """
    Compute covariance matrices for all calsses.

    Parameters
    ----------
    x : 2D List
        The training data.
    y : 2D List
        The training labels.
    classes : List
        List of possible classes.

    Returns
    -------
    covs : 3D List (classes,features,features)
        All covariance matrices for all calsses.
    """

    covs = []
    for this_class in classes:
        cov = getCovarianceMatrix(X[y == this_class])
        covs.append(cov)
        
    return np.stack(covs)

def getProjectionMatrix(X, y, classes, k = 7):
    """
    Compute CSP projection matrix W.

    Parameters
    ----------
    x : 2D List
        The training data.
    y : 2D List
        The training labels.
    classes : List
        List of possible classes.
    k : int
        Number of components to be selected after CSP transformation.

    Returns
    -------
    W : 2D List
        CSP projection matrix
    """

    covs = getAllCovarianceMatrices(X, y, classes)

    eigenValues,eigenVectors = decomposeMatrix(covs)

    # Sort the eigen values and return the indexes of eigen values in a specific threshold range.
    indexes = np.argsort(np.abs(eigenValues - 0.5))[::-1]
    # Pick the corresponding eigen vectors wrt the chosen eigen values. 
    eigenVectors = eigenVectors[:, indexes]

    # Pick the first K effictive component. 
    # filters = eigenVectors.T
    # pick_filters = filters[: k]

    W = eigenVectors.T[: k]

    return W

def CSPTransformation(data, W):
    """
    Transform data to the new better domain using CSP projection matrix.

    Parameters
    ----------
    data : 2D List
        The data to be transformed.
    W : 2D List
        CSP projection matrix.

    Returns
    -------
    features : 2D List
        The final transformed features
    """
    Z = ComputeZ(W, data)

    features = getFeatureVector(Z)

    return features

def extractTopFeatures(filtered_data, bands, weights, mibif_indexes):
    """
    Transform data to the new better domain using CSP projection matrix, 
        and select the most important features using the indexes learnt by MIBIF.

    Parameters
    ----------
    filtered_data : 2D List
        The EEG data after filtering through the given bands.
    bands : list
        A list of pass bands that were used in filtering the input data.   
    weights : 2D List
        The learnt CSP projection matrix.
    mibif_indexes : 2D List
        The indexes of the most important features, learnt through MIBIF.

    Returns
    -------
    top_features : 2D List
        The most important features after transforming with CSP and selecting with MIBIF indexes.
    """

    for index, band in enumerate(bands):
        transformed_band_data = CSPTransformation(filtered_data[band], weights[index])

        if index == 0:
            transformed_data = transformed_band_data
        else:
            transformed_data = np.hstack((transformed_data, transformed_band_data))
            
    top_features = transformed_data[:, mibif_indexes]

    return top_features

def extract_CSP(eeg, labels, classes, bands, n_components=7):
    """
    Organize all CSP extraction process for each band.

    Parameters
    ----------
    eeg  : 2D List
        The data to be transformed.
    labels : 2D List
        The training data trials.
    classes : List
        List of all possible values of classes
    bands : List
        List of all bands ranges
    n_components : int
        Number of features to select after CSP.

    Returns
    -------
    csp_train : 2D List
        New tranformed training features after CSP tranformation.
    weights : 2D List
        The projection matrices for each band.
    rearranged_labels : 2D List
        Training data labels after rearranging.
    """
    rearranged_labels = np.concatenate((labels[labels==classes[0]], labels[labels==classes[1]]))
    weights = []

    for index, band in enumerate(bands):
        left = eeg[band][labels == classes[0]]
        right = eeg[band][labels == classes[1]]
        epochs_data = np.concatenate((left, right))

        W = getProjectionMatrix(epochs_data, rearranged_labels, classes, n_components)
        X_train = CSPTransformation(epochs_data, W)

        if index == 0:
            csp_train = X_train
        else:
            csp_train = np.hstack((csp_train, X_train))
        weights.append(W)

    return csp_train, weights, rearranged_labels


# Not used 
def extractDWT(data, type):
    """
    Apply Discrete Wave Transformation (DWT) on the signals to detect known patterns.

    Parameters
    ----------
    data : 2D List
        The signal to be processed.
    type : string
        Name or type of DWT signal used to apply on the data.

    Returns
    -------
    dwt_features : 2D List
        The new extracted features.
    """

    dwt_features = []

    for trial in data:
        (cA, cD) = pywt.dwt(trial, type)
        
        dwt_features.append(cA)

    return dwt_features

