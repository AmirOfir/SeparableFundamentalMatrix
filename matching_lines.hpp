/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#ifndef _OPENCV_MATCHING_POINTS_H_
#define _OPENCV_MATCHING_POINTS_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "np_cv_imp.hpp"

using namespace cv;
using namespace std;

struct line_info
{
    vector<int> matching_indexes;
    Point3f line_eq_abc;
    Point3f line_eq_abc_norm;
    Point2f bottom_left_edge_point;
    Point2f top_right_edge_point;
    float max_distance;
    int line_index;
};

void FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
    const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
    int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio);

#endif // !_OPENCV_MATCHING_POINTS_H_
