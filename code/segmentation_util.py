"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import segmentation as seg
from scipy import ndimage, stats
import segmentation_project as pro

def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    #------------------------------------------------------------------#
    # Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter

    g = np.zeros_like(x);
    for k in range(x.size):
        xh1=x.copy()
        xh2=x.copy()
        xh1[k]=xh1[k]+h/2
        xh2[k] = xh2[k] - h / 2
        a = fun(xh1)
        b = fun(xh2)
        if isinstance(a,tuple):
            g[k]=(a[0]-b[0])/h
        else:
            g[k]=(a-b)/h

    #------------------------------------------------------------------#

    return g

def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scater_data displays a scatterplot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y

    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx,:]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'X, class '+str(i)
        ax.scatter(X[idx2,feature0], X[idx2,feature1], color=c, label=lbl)

    return ax


def create_dataset(image_number, slice_number, task):
    # create_dataset Creates a dataset for a particular subject (image), slice and task
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    # Output:
    # X           - Nxk feature matrix, where N is the number of pixels and k is the number of features
    # Y           - Nx1 vector with labels
    # feature_labels - kx1 cell array with descriptions of the k features

    #Extract features from the subject/slice
    X, feature_labels = extract_features(image_number, slice_number)

    #Create labels
    Y = create_labels(image_number, slice_number, task)

    return X, Y, feature_labels


def extract_features(image_number, slice_number):
    # extracts features for [image_number]_[slice_number]_t1.tif and [image_number]_[slice_number]_t2.tif
    # Input:
    # image_number - Which subject (scalar)
    # slice_number - Which slice (scalar)
    # Output:
    # X           - N x k dataset, where N is the number of pixels and k is the total number of features
    # features    - k x 1 cell array describing each of the k features

    base_dir = '../data/dataset_brains/'

    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif')
    t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif')

    n = t1.shape[0]
    features = ()

    t1f = t1.flatten().T.astype(float)
    t1f = t1f.reshape(-1, 1)
    t2f = t2.flatten().T.astype(float)
    t2f = t2f.reshape(-1, 1)

    X = np.concatenate((t1f, t2f), axis=1)

    features += ('T1 intensity',)
    features += ('T2 intensity',)

    #------------------------------------------------------------------#
    # Extract more features and add them to X.
    # Don't forget to provide (short) descriptions for the features

    I_blurred = ndimage.gaussian_filter(t1, sigma=10)
    X2 = I_blurred.flatten().T
    t1_blur_10 = X2.reshape(-1, 1)
    features += ('T1 blur sigma=10',)

    I_blurred_t2 = ndimage.gaussian_filter(t2, sigma=10)
    X3 = I_blurred_t2.flatten().T
    t2_blur_10 = X3.reshape(-1, 1)
    features += ('T2 blur sigma=10',)



    t_diff= (t1-t2)^2
    tf_diff = t_diff.flatten().T.astype(float)
    tf_diff = tf_diff.reshape(-1, 1)
    features += ('diff t1-t2',)

    t1_med = pro.create_my_feature(t1)
    t1f_med = t1_med.flatten().T
    t1f_med = t1f_med.reshape(-1, 1)
    features += ('T1 median filter size=9',)

    t2_med = pro.create_my_feature(t2)
    t2f_med = t2_med.flatten().T
    t2f_med = t2f_med.reshape(-1, 1)
    features += ('T2 median filter size=9',)

    c,c_im = seg.extract_coordinate_feature(t1)
    features += ('Center',)

    X = np.concatenate((X, t1_blur_10, t2_blur_10, tf_diff, t1f_med, t2f_med, c),axis=1)
    #------------------------------------------------------------------#
    return X, features


def create_labels(image_number, slice_number, task):
    # Creates labels for a particular subject (image), slice and
    # task
    #
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    #
    # Output:
    # Y           - Nx1 vector with labels
    #
    # Original labels reference:
    # 0 background
    # 1 cerebellum
    # 2 white matter hyperintensities/lesions
    # 3 basal ganglia and thalami
    # 4 ventricles
    # 5 white matter
    # 6 brainstem
    # 7 cortical grey matter
    # 8 cerebrospinal fluid in the extracerebral space

    #Read the ground-truth image
    base_dir = '../data/dataset_brains/'

    I = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_gt.tif')

    if task == 'brain':
        Y = I>0
    elif task == 'tissue':
        white_matter = (I == 2) | (I == 5)
        gray_matter  = (I == 7) | (I == 3)
        csf         = (I == 4) | (I == 8)
        background  = (I == 0) |  (I == 1) | (I == 6)

        Y = np.copy(I)

        Y[background] = 0
        Y[white_matter] = 1
        Y[gray_matter] = 2
        Y[csf] = 3
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1,1)

    return Y


def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # returns the Dice coefficient for two binary label vectors
    # Input:
    # true_labels         Nx1 binary vector with the true labels
    # predicted_labels    Nx1 binary vector with the predicted labels
    # smooth              smoothing factor that prevents division by zero
    # Output:
    # dice          Dice coefficient

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    # Implement the missing functionality for Dice overlap
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(p)):
        if t[i] == p[i] == 1:
            TP += 1
        if p[i] == 1 and t[i] != p[i]:
            FP += 1
        if t[i] == p[i] == 0:
            TN += 1
        if p[i] == 0 and t[i] != p[i]:
            FN += 1
    dice = 2*TP/(2*TP+FP+FN)
    #------------------------------------------------------------------#
    return dice


def dice_multiclass(true_labels, predicted_labels):
    #dice_multiclass.m returns the Dice coefficient for two label vectors with
    #multiple classses
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # dice_score          Dice coefficient

    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    #Consider each class as the foreground class
    for i in np.arange(len(all_classes)):
        idx2 = indices2 == all_classes[i]
        lbl = 'X, class '+ str(all_classes[i])
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1  #Class i is foreground
        temp_true[true_labels != all_classes[i]] = 0  #Everything else is background

        temp_predicted = predicted_labels.copy();
        print(temp_predicted.dtype)
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0
        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))

    dice_score_mean = dice_score.mean()

    return dice_score_mean


def classification_error(true_labels, predicted_labels):
    # classification_error.m returns the classification error for two vectors
    # with labels
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # error         Classification error

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    # Implement the missing functionality for classification error
    err_count=np.sum(t != p)
    err=err_count/len(t)
    #------------------------------------------------------------------#
    return err





