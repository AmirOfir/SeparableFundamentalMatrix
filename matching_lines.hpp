// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_MATCHING_LINES_H_
#define _OPENCV_MATCHING_LINES_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "np_cv_imp.hpp"
#include "matching_points.hpp"
#include "line_homography.hpp"

namespace cv { namespace separableFundamentalMatrix
{
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

    class top_line
    {
        void init(const top_line &o)
        {
            num_inliers = o.num_inliers;
            line1_index = o.line1_index;
            line2_index = o.line2_index;
            max_dist = o.max_dist;
            min_dist = o.min_dist;
            homg_err = o.homg_err;
        }
    public:
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
        /*
        top_line() { }
        top_line(const top_line &o) : line_points_1(o.line_points_1), line_points_2(o.line_points_2), 
            inlier_selected_index(o.inlier_selected_index), selected_line_points1(o.selected_line_points1),
            selected_line_points2(o.selected_line_points2) // copy constructor
        {
            init(o);
        }
        top_line(const top_line &&o) noexcept // move constructor
        {
            inlier_selected_index = move(o.inlier_selected_index);
            selected_line_points1 = move(o.selected_line_points1);
            selected_line_points2 = move(o.selected_line_points2);
            line_points_1 = move(o.line_points_1);
            line_points_2 = move(o.line_points_2);
            init(o);
        }
        top_line &operator=(const top_line &o) // Copy assignment
        {
            if (this != &o) {
                inlier_selected_index = move(o.inlier_selected_index);
                selected_line_points1 = move(o.selected_line_points1);
                selected_line_points2 = move(o.selected_line_points2);
                line_points_1 = move(o.line_points_1);
                line_points_2 = move(o.line_points_2);
                init(o);
            }
            return *this;
        }
        top_line &operator=(top_line &&o) // Move assignment
        {
            if (this != &o) {
                inlier_selected_index = move(o.inlier_selected_index);
                selected_line_points1 = move(o.selected_line_points1);
                selected_line_points2 = move(o.selected_line_points2);
                line_points_1 = move(o.line_points_1);
                line_points_2 = move(o.line_points_2);
                init(o);
            }
            return *this;
        }*/
    };

    vector<top_line> FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
        const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
        int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio);
}}


#endif // !_OPENCV_MATCHING_LINES_H_
