import cv2
import numpy as np
import math
import imutils

class Stitch:
    def __init__(self,img_arr):

        center = int(len(img_arr)/2)
        self.reference_img = img_arr.pop(center)
        self.img_arr = img_arr

        self.sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5), 0)
        kp, des = self.sift.detectAndCompute(gray,None)
        return kp, des

    def match(self, des1, des2):
        matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        rawMatches = matcher.knnMatch(des2,des1,k=2)

        good_matches = []
        ratio_thresh = 0.85

        for x, y in rawMatches:
            if x.distance < y.distance * ratio_thresh:
                good_matches.append(x)

        return good_matches

    def homography(self, kp1, kp2, matches):
        kp1 = [kp1[m.trainIdx] for m in matches]
        kp2 = [kp2[m.queryIdx] for m in matches]

        pts1 = np.array([k.pt for k in kp1])
        pts2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        inlierRatio = float(np.sum(status)) / float(len(status))

        H = H / H[2,2]
        H_inv = np.linalg.inv(H)

        return H, H_inv, inlierRatio

    def find_shape(self, img, H_inv):
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


        input_img_warp = input_img_warp.astype('uint8')

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

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,3)

        ret,thresh = cv2.threshold(gray,1,255,0, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
        roi = cv2.boundingRect(contours[0])

        img = img[roi[1]:roi[3], roi[0]:roi[2]]

        return img

    def crop(self, img):

        final_result = cv2.copyMakeBorder(img, 10, 10 ,10 ,10,
                    cv2.BORDER_CONSTANT, (0,0,0))
        # Convert ảnh sang gray để chia ảnh
        gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Chia ảnh thành black - white (white là panorama)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]

        # Detect Cạnh viền
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key = cv2.contourArea)

        mask = np.zeros(thresh.shape, dtype = 'uint8')
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:

            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        final_result = final_result[y:y + h, x:x + w]

        return final_result

    def stitch(self):

        reference_img = self.reference_img

        while len(self.img_arr) > 0:

            input_img = self.img_arr.pop(0)

            kp1, des1 = self.get_features(reference_img)
            kp2, des2 = self.get_features(input_img)

            matches = self.match(des1, des2)

            H, H_inv, inlierRatio = self.homography(kp1, kp2, matches)

            if inlierRatio > 0.1:

                final_img = self.image_align(reference_img, input_img, H_inv)

                reference_img = final_img

        return reference_img


    def fit_transform(self):
        output = self.stitch()
        cv2.imwrite('output.jpg', output)
        return output
