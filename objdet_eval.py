# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import misc.objdet_tools as tools
import sys
import os
from operator import itemgetter
from shapely.geometry import Polygon
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# change backend so that figure maximizing works on Mac as well
# matplotlib.use('wxagg')


# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):

    # find best detection for each valid label
    true_positives = 0  # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # exclude all labels from statistics which are not considered valid

            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######
            #######
            print("student task ID_S4_EX1 ")

            # step 1 : extract the four corners of the current label bounding-box
            BB_label = tools.compute_box_corners(
                label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.box.heading)
            # step 2 : loop over all detected objects
            for obj in detections:
                # step 3 : extract the four corners of the current detection
                BB_detection = tools.compute_box_corners(
                    obj[1], obj[2], obj[5], obj[6], obj[7])

                # step 4 : compute the center distance between label and detection bounding-box in x, y, and z
                dis_cent_x = label.box.center_x - obj[1]
                dis_cent_y = label.box.center_y - obj[2]
                dis_cent_z = label.box.center_z - obj[3]
                # step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                IoU = (Polygon(BB_label).intersection(Polygon(BB_detection)).area) / \
                    (Polygon(BB_label).union(Polygon(BB_detection)).area)
                # step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if IoU > min_iou:
                    matches_lab_det.append(
                        [IoU, dis_cent_x, dis_cent_y, dis_cent_z])
                    true_positives += 1
            #######
            ####### ID_S4_EX1 END #######

        # find best match and compute metrics
        if matches_lab_det:
            # retrieve entry with max iou in case of multiple candidates
            best_match = max(matches_lab_det, key=itemgetter(1))
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ####### ID_S4_EX2 START #######
    #######
    print("student task ID_S4_EX2")

    # compute positives and negatives for precision/recall

    # step 1 : compute the total number of positives present in the scene
    all_positives = len(labels_valid)

    # step 2 : compute the number of false negatives
    false_negatives = all_positives - true_positives

    # step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives

    #######
    ####### ID_S4_EX2 END #######

    pos_negs = [all_positives, true_positives,
                false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]

    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all, configs):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    ####### ID_S4_EX3 START #######
    #######
    print('student task ID_S4_EX3')

    # step 1 : extract the total number of positives, true positives, false negatives and false positives
    all_positives_sum, true_positives_sum, false_negatives_sum, false_positives_sum = np.sum(
        np.array(pos_negs), 0)
    # step 2 : compute precision
    precision = true_positives_sum / (true_positives_sum + false_positives_sum)

    # step 3 : compute recall
    recall = true_positives_sum / (true_positives_sum + false_negatives_sum)

    #######
    ####### ID_S4_EX3 END #######
    print('precision = ' + str(precision) + ", recall = " + str(recall))

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    # std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union',
              'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (
                     np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (
                     np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()
