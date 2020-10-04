# import cv2
# import sepfm
# a = cv2.imread('a.jpg')
# b = cv2.imread('a.jpg')
# sepfm.findMat(a,b,1,2)

import numpy as np
import cv2
import sepfm
from separable_F_utils import  output_results,get_matched_points

# Last parameter is the scaling factor
# Set this parameter if the results are not satisfactory enough on new image pairs.
# imgs_name={1:('im1a.jpg','im1b.jpg',0.5)}
imgs_name={1:('im1a.jpg', 'im1b.jpg',0.5),
           2:('im2a.jpg', 'im2b.jpg', 1),
           3:('im3a.jpg', 'im3b.jpg', 1),
           4:('im4a.png' ,'im4b.png',0.3),
           5:('im5a.jpg' ,'im5b.jpg',0.75)}

# What is an inlier?
inlier_threshold=3
# How many iterations by default
inlier_ratio    =0.4

print("Number of iterations for {} inliers rate: RANSAC is {} Ours is  {}".format(inlier_ratio,
         int(np.log(0.01) / np.log(1 - inlier_ratio ** 8))+1,
      2*int((np.log(0.01) / np.log(1 - inlier_ratio ** 5)+1)))) # Here we use at most top two lines so we take twice the number, just in case

for img_num in range(1,len(imgs_name)+1):
    print("Image pairs:" + str(img_num))
    pts1,pts2,img1c,img2c,_,_=get_matched_points(*list(imgs_name[img_num][0:3]))

    F8ransac, inliers_ransac = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    #  F_ELH[0] is RANSAC, F_ELH[1] is LMEDS
    F_ELH = sepfm.findMat(pts1, pts2, np.shape(img1c)[0],np.shape(img1c)[1])
    print(F_ELH)
    output_results(img_num, F_ELH, None, F8ransac, img1c, img2c, pts1, pts2, inlier_threshold)
