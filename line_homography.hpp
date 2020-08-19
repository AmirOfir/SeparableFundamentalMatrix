// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_LINE_HOMOGRAPHY_H_
#define _OPENCV_LINE_HOMOGRAPHY_H_

#include <opencv2/opencv.hpp>
#include "np_cv_imp.hpp"

namespace cv {
    namespace separableFundamentalMatrix {
        Mat findLineHomography(const vector<Vec4f> &points);
        void normalizeCoordinatesByLastCol(InputArray _src, OutputArray _dst);
        void lineRansac(int numIterations, const vector<Point2f> &matchingPoints1, const vector<Point2f> &matchingPoints2, float inlierTh = 0.35);
        Mat lineHomographyError(Mat model, const vector<Vec4f> &data);
    }
}

#endif
