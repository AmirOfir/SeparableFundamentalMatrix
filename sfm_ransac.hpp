// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_SFM_RANSAC_H_
#define _OPENCV_SFM_RANSAC_H_

#include <opencv2/opencv.hpp>
#include "matching_lines.hpp"

namespace cv
{
    namespace separableFundamentalMatrix
    {
        struct FastFundamentalMatrixRansacResult
        {

        };

        
        void h_coordinates(InputOutputArray _pts);
        Mat prepareDataForRansac(InputArray _pts1, InputArray _pts2);
        vector<Mat> prepareLinesForRansac(const vector<top_line> &topMatchingLines);
        FastFundamentalMatrixRansacResult fastFundamentalMatrixRansac(uint maxNumIterations, Mat data, 
            int inlier_threshold, const vector<Mat> &arrlines);
    }
}



#endif // !_OPENCV_SFM_RANSAC_H_

