import cv2
import numpy as np
import os
import math
import imutils

# img_path = os.path.split('./images/test3/S3.jpg')
# img_dir = os.listdir(img_path[0])
# img_dir = map(lambda x: os.path.join(img_path[0], x), img_dir)
# img_dir = list(filter(lambda x: x[-6:]!= img_path[1], img_dir))
# print(img_dir)

fpath = './images/test4'
img_dir = os.listdir(fpath)
img_dir = [os.path.join(fpath, file) for file in img_dir]
center = int(len(img_dir)/2)
base_img = img_dir[center]
img_dir.pop(center)
# print(img_dir)


base_img = cv2.imread(base_img)
sift = cv2.xfeatures2d.SIFT_create()

# Keypoint Matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)



while len(img_dir) > 0:

    kp1, des1 = sift.detectAndCompute(base_img,None)

    next_img = cv2.imread(img_dir[0])
    kp2, des2 = sift.detectAndCompute(next_img,None)

    match = cv2.FlannBasedMatcher(index_params, search_params)
    rawMatches = match.knnMatch(des2,des1,k=2)

    good_matches = []
    ratio_thresh = 0.85

    for x, y in rawMatches:
        if x.distance < y.distance * ratio_thresh:
            good_matches.append(x)

    # # Distance from key image
    # sumDistance = 0.0
    #
    # for match in good_matches:
    #     sumDistance += match.distance
    #
    # meanPointDistance = sumDistance/float(len(good_matches))

    kp1 = [kp1[m.trainIdx] for m in good_matches]
    kp2 = [kp2[m.queryIdx] for m in good_matches]

    pts1 = np.array([k.pt for k in kp1])
    pts2 = np.array([k.pt for k in kp2])

    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # print(np.sum(status), len(status))

    inlierRatio = float(np.sum(status)) / float(len(status))

    img_dir.pop(0)

    H = H / H[2,2]
    H_inv = np.linalg.inv(H)

    if inlierRatio > 0.1:
        # Find Dimmension:
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        y,x = next_img.shape[:2]

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

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, base_img.shape[1])
        max_y = max(max_y, base_img.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if ( min_x < 0 ):
            move_h[0,2] += -min_x
            max_x += -min_x

        if ( min_y < 0 ):
            move_h[1,2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * H_inv

        # "Min Points: ", (min_x, min_y)
        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        # "New Dimensions: ", (img_w, img_h)

        # crop edges
        base_h, base_w, base_d = base_img.shape
        next_h, next_w, next_d = next_img.shape

        base_img = base_img[5:base_h-5, 5:base_w-5]
        next_img = next_img[5:next_h-5, 5:next_w-5]

        # Warp the new image given the homography from the old image
        base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h))

        next_img_warp = cv2.warpPerspective(next_img, mod_inv_h, (img_w, img_h))

        # cv2.imshow('base', cv2.resize(base_img_warp,(500,500)))
        # cv2.imshow('next', cv2.resize(next_img_warp,(500,500)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Put the base image on an enlarged palette
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        # Create masked composite
        (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_BINARY)

        # cv2.imshow('output', np.hstack((base_img_warp[:,:img_w], next_img_warp[:,img_w]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # add base image
        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
            mask=np.bitwise_not(data_map),dtype=cv2.CV_8U)

        # add next image
        result = cv2.add(enlarged_base_img, next_img_warp,dtype=cv2.CV_8U)

        # Crop black edge
        final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print "Found %d contours..." % (len(contours))

        max_area = 0
        best_rect = (0,0,0,0)

        cnt = imutils.grab_contours(contours)
        cnt = max(cnt, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(cnt)

        deltaHeight = h-y
        deltaWidth = w-x

        area = deltaHeight * deltaWidth

        if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
            max_area = area
            best_rect = (x,y,w,h)

        if ( max_area > 0 ):
            final_img_crop = result[best_rect[1]:best_rect[1]+best_rect[3],
                    best_rect[0]:best_rect[0]+best_rect[2]]

            final_img = final_img_crop

        base_img = final_img

        rows, cols = np.where(base_img[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        base_img = base_img[min_row:max_row, min_col:max_col, :]




# cv2.imshow('test', result)
cv2.imwrite('output2.jpg', base_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
