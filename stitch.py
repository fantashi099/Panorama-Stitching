import cv2
import numpy as np
import math
import imutils

class Stitch:
    def __init__(self,img_arr):

        """
        For the better performance and the least distorted, I start from
        the center of images. Input of class is numpy array images input
        because of output from web server.
        """

        center = int(len(img_arr)/2)
        self.reference_img = img_arr.pop(center)
        self.img_arr = img_arr

        # Generate SIFT and Fast Library for Approximate Nearest Neighbors
        self.sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)

    def get_features(self, img):
        """
        Using sift to detect key point features and descriptors.
        The key point usually contains the patch 2D position and
        other stuff if available such as scale and orientation of
        the image feature.
        The descriptor contains the visual description of the patch
        and is used to compare the similarity between image features.

        Return keypoint and descriptor
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5), 0)
        kp, des = self.sift.detectAndCompute(gray,None)
        return kp, des

    def match(self, des1, des2):
        """
        Matching two images using FLANN with Knn matching between the
        two feature vector sets using k=2.
        Return correct matches
        """

        # FLann fast with less compute to approximate distance points
        matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        rawMatches = matcher.knnMatch(des2,des1,k=2)

        # David Lowe's method for filtering keypoint matches by eliminating matches
        # https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
        good_matches = []
        ratio_thresh = 0.85

        for x, y in rawMatches:
            if x.distance < y.distance * ratio_thresh:
                good_matches.append(x)

        return good_matches

    def homography(self, kp1, kp2, matches):
        """
        Estimate homography between pairs of two images by keypoint
        Return Homography matrix and it inverted and inlier/matched
        """

        # Estimate Homography matrix
        kp1 = [kp1[m.trainIdx] for m in matches]
        kp2 = [kp2[m.queryIdx] for m in matches]

        pts1 = np.array([k.pt for k in kp1])
        pts2 = np.array([k.pt for k in kp2])

        # Using RANSAC algorithm with aximum pixel “wiggle room” allowed is 5
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        # RANSAC is an iterative algorithm for the robust estimation of parameters
        # from a subset of inliers from the complete data set.
        inlierRatio = float(np.sum(status)) / float(len(status))

        H = H / H[2,2]
        H_inv = np.linalg.inv(H)

        return H, H_inv, inlierRatio

    def find_shape(self, img, H_inv):
        """
        Find new panorama shape with adjusted corners to cover two warping images
        """

        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        y,x = img.shape[:2]

        base_p1[:2] = [0,0]
        base_p2[:2] = [x,0]
        base_p3[:2] = [0,y]
        base_p4[:2] = [x,y]

        max_x = None
        max_y = None
        min_x = None
        min_y = None

        for pts in [base_p1, base_p2, base_p3, base_p4]:
            # Find min max x,y : 4 coordiante shape
            hp = np.matrix(H_inv, np.float32) * np.matrix(pts, np.float32).T

            hp_arr = np.array(hp, np.float32)

            normal_pts = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

            if ( max_x == None or normal_pts[0,0] > max_x ):
                max_x = normal_pts[0,0]

            if ( max_y == None or normal_pts[1,0] > max_y ):
                max_y = normal_pts[1,0]

            if ( min_x == None or normal_pts[0,0] < min_x ):
                min_x = normal_pts[0,0]

            if ( min_y == None or normal_pts[1,0] < min_y ):
                min_y = normal_pts[1,0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return min_x, min_y, max_x, max_y

    def image_align(self, reference_img, input_img, H_inv):
        """
        Perform Stitch warping two images with the given homography and stitch together
        Return new Stitched image
        """

        min_x, min_y, max_x, max_y = self.find_shape(input_img, H_inv)

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, reference_img.shape[1])
        max_y = max(max_y, reference_img.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if ( min_x < 0 ):
            move_h[0,2] += -min_x
            max_x += -min_x

        if ( min_y < 0 ):
            move_h[1,2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * H_inv

        # "Min Points: ", (min_x, min_y)
        # "New Dimensions: ", (img_w, img_h)
        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        # crop edges
        base_h, base_w, base_d = reference_img.shape
        next_h, next_w, next_d = input_img.shape

        reference_img = reference_img[5:base_h-5, 5:base_w-5]
        input_img = input_img[5:next_h-5, 5:next_w-5]

        # Warp the new image given the homography from the old image
        reference_img_warp = cv2.warpPerspective(reference_img, move_h, (img_w, img_h))
        input_img_warp = cv2.warpPerspective(input_img, mod_inv_h, (img_w, img_h))

        # Put the base image on an enlarged palette
        enlarged_reference_img = np.zeros((img_h, img_w, 3), np.uint8)

        # Create masked composite
        (ret,data_map) = cv2.threshold(cv2.cvtColor(input_img_warp, cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_BINARY)

        # add base image
        enlarged_reference_img = cv2.add(enlarged_reference_img, reference_img_warp,
            mask=np.bitwise_not(data_map),dtype=cv2.CV_8U)

        # add next image
        result = cv2.add(enlarged_reference_img, input_img_warp,dtype=cv2.CV_8U)

        final_result = self.remove_seam(result)

        return final_result


    def remove_seam(self, img):
        """
        Remove black region after panorama stitching
        Return panorama image with 4 coordiante (miny,minx) (max y, minx) (miny,max x) (max y, max x) of image in contours
        """

        # https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,3)

        ret,thresh = cv2.threshold(gray,1,255,0, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contours of the image, then cropping it
        contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
        # Return 4 coordiante of contours
        roi = cv2.boundingRect(contours[0])

        # Cropping
        img = img[roi[1]:roi[3], roi[0]:roi[2]]

        return img

    def crop(self, img):
        """
        Cropping the main features of image
        """

        final_result = cv2.copyMakeBorder(img, 10, 10 ,10 ,10,
                    cv2.BORDER_CONSTANT, (0,0,0))
        # Convert to gray scale to thresh binary
        gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Convert image into black - white (white is panorama)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]

        # Detect contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key = cv2.contourArea)

        # allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype = 'uint8')
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # create two copies of the mask: one to serve as our actual
		# minimum rectangular region and another to serve as a counter
		# for how many pixels need to be removed to form the minimum
		# rectangular region
        minRect = mask.copy()
        sub = mask.copy()

        # keep looping until there are no non-zero pixels left in the subtracted image
        while cv2.countNonZero(sub) > 0:

            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        # find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
        contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

        # Cropping
        final_result = final_result[y:y + h, x:x + w]

        return final_result

    def stitch(self):
        """
        Assume that images array was supplied in order, choose middle image as the
        reference image and the first image of array as the new input image. Use 𝐾NN
        algorithm to search matching feature points between the new input image and
        the reference image in accordance with the minimum Euclidean distance.

        According to the matching feature points data set, use the RANSAC algorithm
        to calculate the affine matrix 𝐻, which can transform the new input image
        into the coordinate space of the reference image.

        Return Final Panorama
        """

        # Use middle image as a reference image
        reference_img = self.reference_img

        # For loop til image array have nothing left
        while len(self.img_arr) > 0:

            input_img = self.img_arr.pop(0)

            # Get features two images
            kp1, des1 = self.get_features(reference_img)
            kp2, des2 = self.get_features(input_img)

            # Matching two descriptor
            matches = self.match(des1, des2)

            # Get homography matrix and RANSAC ratio
            H, H_inv, inlierRatio = self.homography(kp1, kp2, matches)

            if inlierRatio > 0.1:

                # reverse the multiplication order
                final_img = self.image_align(reference_img, input_img, H_inv)

                # Update stitched image to reference image
                reference_img = final_img

        return reference_img

    def fit_transform(self):
        """Do all module of class"""
        output = self.stitch()
        cv2.imwrite('demo/outputCrop5.jpg', self.crop(output))
        cv2.imwrite('demo/outputRaw5.jpg', output)
        return output
