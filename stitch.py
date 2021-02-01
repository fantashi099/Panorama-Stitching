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


    def detect_keypoint(self):
        """I using SIFT to detect interest points and then compute keypoint
        descriptors. Then use Fast Library for Approximate Nearest Neighbors
        for higher speed performance Matching Keypoins."""

        # Initialize first image of array and output of matching-images
        kp = [[None] for i in range(len(self.img_arr))]
        des = [[None] for i in range(len(self.img_arr))]
        kp[0], des[0] = self.sift.detectAndCompute(self.img_arr[0],None)
        img_detect1 = cv2.drawKeypoints(self.img_arr[0],kp[0],None)
        img_match = None

        # For the next images, matching with first image
        for i in range(1,len(self.img_arr)):

            # Using KNN to Approximate Nearest Neighbors
            kp[i], des[i] = self.sift.detectAndCompute(self.img_arr[i],None)
            matcher = cv2.FlannBasedMatcher(self.index_params_FLANN, self.search_params_FLANN)
            rawMatches = matcher.knnMatch(des[i-1],des[i], k=2)
            good_matches = []

            for x, y in rawMatches:
                if x.distance < y.distance * self.ratio_thresh_FLANN:
                    good_matches.append(x)

            good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=True)
            good_matches = good_matches[:200]

            # For returning img_match if required by web user
            img_match = cv2.drawMatches(self.img_arr[i-1], kp[i-1], self.img_arr[i], kp[i], good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return img_match, kp, des, good_matches

    def matching(self, kp, des, good_matches):
        """
        I compute estimate homography matrix to merge images by transforming.
        """
        # _, kp, des, good_matches = self.detect_keypoint(self.img_arr)
        h = [self.img_arr[i].shape[0] for i in range(len(self.img_arr))]
        w = [self.img_arr[i].shape[1] for i in range(len(self.img_arr))]
        max_h = max(h)

        # Estimate Homography matrix and Transform Image
        kp = [np.float32([kpz.pt for kpz in kp[i]]) for i in range(len(kp))]
        pts = [[None] for i in range(len(kp))]

        # Initialize first point
        pts[0] = np.float32([kp[0][m.queryIdx] for m in good_matches])
        for i in range(1,len(self.img_arr)):
            
            pts[i] = np.float32([kp[i][m.trainIdx] for m in good_matches])

            H, status = cv2.findHomography(pts[0], pts[i], cv2.RANSAC)

            self.img_arr[0] = cv2.warpPerspective(self.img_arr[0], H, (w[0] + w[i], max_h))
            self.img_arr[0][0:h[i], 0:w[i]] = self.img_arr[i]

        return self.img_arr[0]

    def blending(self, result):

        rows, cols = np.where(result[:,:,0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result
