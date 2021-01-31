import cv2
import numpy as np
import sys
import imutils

class Stitch:
    def __init__(self, img_arr, ratio_thresh = 0.85):
        self.img_arr = img_arr
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.index_params_FLANN = dict(algorithm = 0, trees = 5)
        self.search_params_FLANN = dict(checks = 50)
        self.ratio_thresh_FLANN = ratio_thresh


    def detect_keypoint(self, img_arr):
        """I using SIFT to detect interest points and then compute keypoint
        descriptors. Then use Fast Library for Approximate Nearest Neighbors
        for higher speed performance Matching Keypoins."""

        # Initialize first image of array and output of matching-images
        kp = [None for i in range(len(img_arr))]
        des = [None for i in range(len(img_arr))]
        kp[0], des[0] = self.sift.detectAndCompute(img_arr[0],None)
        img_detect1 = cv2.drawKeypoints(img_arr[0],kp[0],None)
        img_match = None

        # For the next images, matching with first image
        for i in range(1,len(img_arr)):

            # Using KNN to Approximate Nearest Neighbors
            kp[i], des[i] = self.sift.detectAndCompute(img_arr[i],None)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            rawMatches = match.knnMatch(des[i-1],des[i], k=2)
            good_matches = []

            for x, y in rawMatches:
                if x.distance < y.distance * self.ratio_thresh_FLANN:
                    good_matches.append(x)

            good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=True)
            good_matches = good_matches[:200]

            # For returning img_match if required by web user
            img_match = cv2.drawMatches(img_arr[i-1], kp[i-1], img_arr[i], kp[i], good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return img_match, kp, des, good_matches

    # def matching(self):
    #     """
    #     I compute estimate homography matrix to merge images by transforming.
    #     """
    #     _, kp, des, good_matches = self.detect_keypoint(self.img_arr)
    #
    #     # Estimate Homography matrix and Transform Image
    #     kp = [np.float32([kp.pt for kp in kp[i]]) for i in range(len(kp))]
    #     pts = [None for i in range(len(kp))]
    #
    #     # Initialize first point
    #     pts[0] =
