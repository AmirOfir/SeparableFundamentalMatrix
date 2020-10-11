import numpy as np
import cv2
from separable_F_utils import  output_results,get_matched_points
import timeit
import time
import gc
repeat_count=10

from sepfm import findFundamentalMatRegular, findSeparableMat

# Last parameter is the scaling factor
# Set this parameter if the results are not satisfactory enough on new image pairs.
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
    print("Image pair number" + str(img_num))

    # CV time
    pts1,pts2,img1c,img2c,_,_=get_matched_points(imgs_name[img_num][0], imgs_name[img_num][1], 1)
    maxIters=int(np.log(0.01) / np.log(1 - inlier_ratio ** 8))+1
    def func_to_measure():
         F8ransac = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99, maxIters)
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    duration  = round(np.mean(duration[1:]),5)
    print('opencv time:', duration)

    # Regular time
    def func_to_measure():
         F8ransac = findFundamentalMatRegular(pts1, pts2)#, cv2.FM_RANSAC, 3, 0.99, maxIters)
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    duration  = round(np.mean(duration[1:]),5)
    print('full ransac:', duration)

    pts1,pts2,img1c,img2c,_,_=get_matched_points(*list(imgs_name[img_num][0:3]))
    def func_to_measure():
        F_ELH = findSeparableMat(pts1, pts2, np.shape(img1c)[0],np.shape(img1c)[1])
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    duration  = round(np.mean(duration[1:]),5)
    print('sepfm:', duration)

    del pts1
    del pts2
    del img1c
    del img2c
    gc.collect()
