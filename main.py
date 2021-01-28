import cv2
import numpy as np

# Read Images
img_src1 = cv2.imread('images/a2.jpg')
img_src2 = cv2.imread('images/a3.jpg')
# img_src1 = cv2.imread('http://www.ic.unicamp.br/~helio/imagens_registro/foto1A.jpg')
# img_src2 = cv2.imread('http://www.ic.unicamp.br/~helio/imagens_registro/foto1B.jpg')


# Detect Keypoint
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_src1,None)
kp2, des2 = sift.detectAndCompute(img_src2,None)

# img_detect1 = cv2.drawKeypoints(img_src1,kp1,img_src1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img_detect2 = cv2.drawKeypoints(img_src2,kp2,img_src2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_detect1 = cv2.drawKeypoints(img_src1,kp1,None)
img_detect2 = cv2.drawKeypoints(img_src2,kp2,None)

# Keypoint Matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
rawMatches = match.knnMatch(des1,des2,k=2)
good_matches = []
ratio_thresh = 0.7

for x, y in rawMatches:
    if x.distance < y.distance * ratio_thresh:
        good_matches.append(x)

good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=True)
good_matches = good_matches[:200]

img_match = cv2.drawMatches(img_src1, kp1, img_src2, kp2, good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Estimate Homography matrix and Transform Image
kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

pts1 = np.float32([kp1[m.queryIdx] for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx] for m in good_matches])

(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)

h1, w1 = img_src1.shape[:2]
h2, w2 = img_src2.shape[:2]

max_h = max(h1,h2)
print(h1,w1)
print(h2,w2)

result = cv2.warpPerspective(img_src1, H, (w1+w2, h1+h2))
result[0:h2, 0:w2] = img_src2

# Blending Image
rows, cols = np.where(result[:, :, 0] != 0)
min_row, max_row = min(rows), max(rows) + 1
min_col, max_col = min(cols), max(cols) + 1
final_result = result[min_row:max_row, min_col:max_col, :]


cv2.imshow('Show Panorama Image', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
