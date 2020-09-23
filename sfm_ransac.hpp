// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_SFM_RANSAC_H_
#define _OPENCV_SFM_RANSAC_H_

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d_c.h>

#include "pointset_registrator.hpp"

namespace cv
{

Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
    int _modelPoints, double _threshold=0, double _confidence=0.99, int _maxIters=1000);

}

#endif // !_OPENCV_SFM_RANSAC_H_

