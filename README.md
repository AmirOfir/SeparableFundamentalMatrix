# Separable Fundamental Matrix

https://arxiv.org/abs/2006.05926 \
http://www.gil-ba.com/sep_f/SeparableF.html 

This repository contains a C++ implementation of the Separable Four Points Fundamental Matrix Finding algorithm. 

## How it works
1. Optionally, resizing the images to a smaller size.
1. Matching line segments with at least 4 points between two images.
2. Running RANSAC to find the longest segments with most inliers, and picking the lines with the max angle.
3. For each line, sampling 4 more points for a RANSAC to compute the fundamental matrix.
4. Returning the matrix with the most inliers
5. Optionally, computing a matrix from the inliers.

## Building and running
1. To build and run on windows using VS 2017, provided a .sln with complete tests and a tester
2. Integration (pending, Oct 20) to opencv is available on https://github.com/AmirOfir/opencv_contrib/tree/Separable_Fundemental_matrix \
Use standard installation using cmake.
3. To build and run on python (tested on windows 10, anaconda, python 3.7.7, opencv 4.1.1): 

        python setup.py install
        
        -- Importing 
        import sepfm
4. For finding a fundamental matrix:

        sepfm.findSeparableMat(img1_points, img2_points, img_height, img_width) 

5. For running opencv code with full RANSAC:

        sepfm.findFundamentalMatRegular(img1_points, img2_points) 

## Testing
1. Tester directory contains a set of images with corresponding point sets in csv format.

        python tester\test.py
2. Time comparing against full RANSAC are available

        python tester\times.py

## Contribution and further requirements
For exposure of additional arguments to python, bug reports, and centra please do not hesitate to reaching us.

## Footnotes
Credits for the full RANSAC are available from opencv.

