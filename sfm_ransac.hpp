// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_SFM_RANSAC_H_
#define _OPENCV_SFM_RANSAC_H_

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include "SFM_finder.hpp"
#include "pointset_registrator.hpp"

namespace cv
{
    namespace separableFundamentalMatrix
    {
        struct FastFundamentalMatrixRansacResult
        {

        };

        
        class SFMRansac
        {
        public:
            static void h_coordinates(InputOutputArray _pts);
            static Mat prepareDataForRansac(InputArray _pts1, InputArray _pts2);
            static vector<tuple<Mat,Mat>> prepareLinesForRansac(const vector<top_line> &topMatchingLines);
            FastFundamentalMatrixRansacResult fastFundamentalMatrixRansac(uint maxNumIterations, Mat data,
                int inlier_threshold, const vector<Mat> &arrlines);
        };

        
    }
}

#endif // !_OPENCV_SFM_RANSAC_H_

