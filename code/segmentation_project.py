#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project code+scripts for 8DC00 course
"""

# Imports

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from scipy import ndimage, stats
import timeit


def create_my_feature(I):

    # This function is used to create median filtered images of both T1 and T2 data.
    # It seems rather trivial; however, the added context and added "grouping" effect
    # seemed quite helpful for a K-nn algorithm to us.
    # Note that there will be no implementation of a PCA in this function, since we didn't find it to be incredibly
    # helpful for the features used in our method.
    #
    # Other features we created involved 1) distance to center and 2) difference between T1 and T2 values

    I_med_filt = ndimage.median_filter(I, 9)

    return I_med_filt


def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='tissue'):
    # segments the image based on your own method!
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # task           String corresponding to the segmentation task: either 'brain' or 'tissue'
    # Output:
    # predicted_labels    Predicted labels for the test slice

    # =====================================================================================

    # This method is comprised of a two steps, being:
    #
    # 1) Segment the entire image using a combined K-nn algorithm (optimised via a learning curve)
    #
    # 2) Perform a morphological opening on this mask, in order to smooth out some of the poorly segmentable skull.
    #    Note that testing with this method during prototyping showed an increase in dice and decrease in error.
    #
    # ########################

    # Set up computing timer
    start_seg = timeit.default_timer()

    # Provide program with feature selection
    feature_sets = [(0,1), (0,2), (0,3), (0,4), (0,7), (4,7)]
    print('\nNumber of feature sets:', len(feature_sets),'...\n')

    # Set up prediction matrix
    r, c = train_labels_matrix.shape

    predicted_labels = np.empty([r, len(feature_sets)])
    predicted_labels[:] = np.nan

    # STEP 1: SEGMENTATION

    predicted_labels_features = predicted_labels.copy()

    for set in range(len(feature_sets)):
        error_list = []

        # Create learning curve and optimise over training data (we naturally can't use test data for optimisation)
        k_list = [1, 3, 5, 7, 9, 11]
        for k in k_list:

            predicted_train_labels = predicted_labels.copy()

            for i in np.arange(c-1):
                predicted_train_labels[:, i] = seg.segmentation_knn(train_data_matrix[:, feature_sets[set], i], train_labels_matrix[:, i], train_data_matrix[:, feature_sets[set], c-1], k)

            # combine labels
            predicted_train_labels = stats.mode(predicted_train_labels, axis=1)[0]
            # compute and store error
            err = util.classification_error(train_labels_matrix[:, c-1], predicted_train_labels)
            error_list.append(err)

        # Find lowest error in list
        optimum = error_list.index(min(error_list))
        k_opt = k_list[optimum]

        # Now, run the final K-nn algorithm for this feature set, for known k.
        for i in np.arange(c):
            predicted_labels[:, i] = seg.segmentation_knn(train_data_matrix[:, :, i], train_labels_matrix[:, i], test_data, k_opt)

        # Store the predicted labels for this feature
        predicted_labels_feat = stats.mode(predicted_labels, axis=1)[0]
        predicted_labels_features[:, set] = predicted_labels_feat[:, 0]

        print('Completed feature set ', set+1)

    # Combine labels
    predicted_labels_v1 = stats.mode(predicted_labels_features, axis=1)[0]
    predicted_mask_v1 = predicted_labels_v1.reshape([240,240])

    # STEP 2: MORPHOLOGICAL OPENING

    # Perform opening. Note that the input image isn't binary. However, the function accounts for this, so not to worry.
    # (num_iter = 2 was chosen via trial and error)
    openimage = ndimage.morphology.binary_opening(predicted_mask_v1, iterations=1)

    # Multiply opened binary mask with original mask
    predicted_mask_v2 = predicted_mask_v1*openimage

    # Flatten newly acquired mask for later use
    predicted_labels_v2 = predicted_mask_v2.flatten()

    predicted_labels = predicted_labels_v2
    # ------------------------------------------------------------------ #
    stop_seg = timeit.default_timer()
    run_time = stop_seg-start_seg

    print("\nCOMPUTING TIME: ",round(run_time),"seconds")

    return predicted_labels


def segmentation_final():
    print("=================== RUN SEGMENTATION_FINAL ===================\n")
    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'tissue'

    #Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    true_mask = test_labels.reshape(240, 240)
    predicted_mask = predicted_labels.reshape(240, 240)

    # fig = plt.figure(figsize=(8,8))
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(true_mask, 'gray')
    # ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    # print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'tissue'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images], dtype=int) # dtype=bool.. That took me about two hours.

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    start = timeit.default_timer()
    print('Finished loading data.\n\nStarting segmentation...\t[T=', round(timeit.default_timer()-start,3),'s]')

    #Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        print('\n\n====== SUBJECT',i+1,'======\t\t[T=',round(timeit.default_timer()-start,3),'s]')
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(15,5))

        predicted_labels = seg.segmentation_combined_atlas(train_labels_matrix)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,0] = util.dice_multiclass(test_labels, predicted_labels)
        predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(131)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,1] = util.dice_multiclass(test_labels, predicted_labels)
        predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(132)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
        all_errors[i,2] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,2] = util.dice_multiclass(test_labels, predicted_labels)
        predicted_mask_3 = predicted_labels.reshape(im_size[0],im_size[1])
        ax3 = fig.add_subplot(133)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,2], all_dice[i,2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))

        print("Dice:",round(all_dice[i,2],3),"; Error:",round(all_errors[i,2],3))

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])

    stop_time = timeit.default_timer()
    print("\n== == == == == == == ==\n\nTOTAL COMPUTING TIME: ",round((stop_time-start)/60,2),"min")

    print("\n=================== END SEGMENTATION_FINAL ===================")

def segmentation_demo():

    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'tissue'

    #Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    true_mask = test_labels.reshape(240, 240)
    predicted_mask = predicted_labels.reshape(240, 240)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(true_mask, 'gray')
    ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'tissue'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    print('Finished loading data.\nStarting segmentation...')

    #Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(15,5))

        predicted_labels = seg.segmentation_combined_atlas(train_labels_matrix)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,0] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(131)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,1] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(132)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
        all_errors[i,2] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,2] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_3 = predicted_labels.reshape(im_size[0],im_size[1])
        ax3 = fig.add_subplot(133)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,2], all_dice[i,2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))
