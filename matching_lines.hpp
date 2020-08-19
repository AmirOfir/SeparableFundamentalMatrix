// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_MATCHING_POINTS_H_
#define _OPENCV_MATCHING_POINTS_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "np_cv_imp.hpp"
#include "line_homography.hpp"

namespace cv { namespace separableFundamentalMatrix
{
    using namespace cv;
    using namespace std;

    struct line_info
    {
        vector<int64> matching_indexes;
        Point3f line_eq_abc;
        Point3f line_eq_abc_norm;
        Point2f bottom_left_edge_point;
        Point2f top_right_edge_point;
        float max_distance;
        int line_index;
    };

    struct top_line
    {
        int num_inliers;
        vector<Point2f> line_points_1;
        vector<Point2f> line_points_2;
        int line1_index;
        int line2_index;
        vector<int> inlier_selected_index;
        vector<Point2f> selected_line_points1;
        vector<Point2f> selected_line_points2;
        float max_dist;
        float min_dist;
        float homg_err;

    };

    void FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
        const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
        int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio);
}}


#endif // !_OPENCV_MATCHING_POINTS_H_
