# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import misc.params as params
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Track:
    '''Track class with state, covariance, id, score'''

    def __init__(self, meas, id):
        print('creating track no.', id)
        # rotation matrix from sensor to vehicle coordinates
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]

        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        self.x = np.matrix([[49.53980697],
                            [3.41006279],
                            [0.91790581],
                            [0.],
                            [0.],
                            [0.]])
        # self.x = np.matrix([[0],
        #                     [0],
        #                     [0],
        #                     [0.],
        #                     [0.],
        #                     [0.]])
        self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                            [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                            [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
                            [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
                            [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
                            [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])
        # self.state = 'confirmed'
        # self.score = 0
        measured_data = np.ones((4, 1))
        measured_data[0:3] = meas.z
        transfered_data = np.matmul(meas.sensor.sens_to_veh, measured_data)
        self.x[0:3] = transfered_data[0:3]
        Rot_Mat = meas.sensor.sens_to_veh[0:3, 0:3]
        self.P[0:3, 0:3] = np.matmul(
            np.matmul(M_rot, meas.R), np.transpose(M_rot))
        self.P[3:6, 3:6] = np.diag(
            [params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])

        self.state = 'initialized'
        self.score = 1./params.window

        ############
        # END student code
        ############

        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        # transform rotation from sensor to vehicle coordinates
        self.yaw = np.arccos(
            M_rot[0, 0]*np.cos(meas.yaw) + M_rot[0, 1]*np.sin(meas.yaw))
        self.t = meas.t

    def set_x(self, x):
        self.x = x

    def set_P(self, P):
        self.P = P

    def set_t(self, t):
        self.t = t

    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            # transform rotation from sensor to vehicle coordinates
            self.yaw = np.arccos(
                M_rot[0, 0]*np.cos(meas.yaw) + M_rot[0, 1]*np.sin(meas.yaw))


###################

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''

    def __init__(self):
        self.N = 0  # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        old_tracks = []
        # decrease score for unassigned tracks
        # print(unassigned_tracks)
        # print("unassigned_meas =")
        # print(unassigned_meas)
        # print("unassigned_tracks =")
        # print(unassigned_tracks)
        # print("meas_list =")
        # print(meas_list)
        # print(meas_list != 0)
        for i in unassigned_tracks:
            track = self.track_list[i]
            # print(track.P[0, 0])
            # print(track.P[1, 1])
            # check visibility
            if meas_list:  # if not empty
                # print("in_frame=" + meas_list[0].sensor.in_fov(track.x) != 0)
                print(meas_list[0].sensor.in_fov(track.x) != 0)
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    track.score = max(0, track.score - 1/params.window)
                    if ((track.score < params.delete_threshold and track.state == 'confirmed')
                            or track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P):
                        old_tracks.append(track)
        # # decrease score for unassigned tracks
        # for i in unassigned_tracks:
        #     track = self.track_list[i]
        #     # check visibility
        #     if meas_list:  # if not empty
        #         # TODO: why only the first measurement
        #         if meas_list[0].sensor.in_fov(track.x):
        #             track.score -= 1./params.window
        #             track.score = max(track.score, 0)

        # # delete old tracks
        # for track in self.track_list:
        #     delete_track = False
        #     if (track.state == 'initialized' or track.state == 'tentative') \
        #             and (track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P) \
        #             or track.score < 0.1:
        #         delete_track = True

        #     if track.state == 'confirmed' and track.score < params.delete_threshold:
        #         delete_track = True

        #     if delete_track:
        #         self.delete_track(track)

        # delete old tracks
        # print(old_tracks)
        for track in old_tracks:
            self.delete_track(track)
        ############
        # END student code
        ############

        # initialize new track with unassigned measurement
        for j in unassigned_meas:
            # only initialize with lidar measurements
            if meas_list[j].sensor.name == 'lidar':
                self.init_track(meas_list[j])
    # def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
    #     ############
    #     # TODO Step 2: implement track management:
    #     # - decrease the track score for unassigned tracks
    #     # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
    #     # feel free to define your own parameters)
    #     ############
    #     # decrease score for unassigned tracks
    #     for i in unassigned_tracks:
    #         track = self.track_list[i]
    #         # check visibility
    #         if meas_list:  # if not empty
    #             if meas_list[0].sensor.in_fov(track.x):
    #                 # your code goes here
    #                 track.score -= 1. / params.window
    #     # delete old tracks
    #     for track in self.track_list:
    #         if ((track.state == 'confirmed' and track.score < params.delete_threshold) or (
    #                 track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P)):
    #             self.delete_track(track)
    #     ############
    #     # END student code
    #     ############
    #     # initialize new track with unassigned measurement
    #     for j in unassigned_meas:
    #         # only initialize with lidar measurements
    #         if meas_list[j].sensor.name == 'lidar':
    #             self.init_track(meas_list[j])

    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)

    def handle_updated_track(self, track):
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############
        track.score += 1./params.window
        track.score = min(1, track.score)
        if track.state == 'initialized' and track.score > params.confirmed_threshold/4:
            track.state = 'tentative'
        # track.score += 1/params.window
        if track.state == 'tentative' and track.score > params.confirmed_threshold:
            track.state = 'confirmed'
    # def handle_updated_track(self, track):
    #     ############
    #     # TODO Step 2: implement track management for updated tracks:
    #     # - increase track score
    #     # - set track state to 'tentative' or 'confirmed'
    #     ############
    #     track.score += 1.0 / params.window
    #     tentative_threshold = 0.2 * params.confirmed_threshold
    #     if (track.score > params.confirmed_threshold):
    #         track.state = 'confirmed'
    #     elif track.state == 'initialized' and track.score > tentative_threshold:
    #         track.state = 'tentative'
    #     if (track.score > 1.0):
    #         track.score = 1.0
        ############
        # END student code
        ############
