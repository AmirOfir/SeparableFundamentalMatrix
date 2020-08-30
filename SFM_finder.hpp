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
#ifndef _OPENCV_SFM_FINDER_H_
#define _OPENCV_SFM_FINDER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "np_cv_imp.hpp"
#include "matching_points.hpp"
#include "line_homography.hpp"

using namespace cv;
using namespace std;

#define DEFAULT_HOUGH_RESCALE -1

namespace cv 
{ 
    namespace separableFundamentalMatrix 
    {
        Mat findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
            float inlier_ratio = 0.4, int inlier_threshold = 3,
            float hough_rescale = DEFAULT_HOUGH_RESCALE, int num_matching_pts_to_use = 150, int pixel_res = 4, int min_hough_points = 4,
            int theta_res = 180, float max_distance_pts_line = 3, int top_line_retries = 2, int min_shared_points = 4);    

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
        };

        class SeparableFundamentalMatFindCommand
        {
        private:
            int imSizeHOrg;
            int imSizeWOrg;
            float inlierRatio; 
            int inlierThreashold; 
            float houghRescale; 
            int numMatchingPtsToUse; 
            int pixelRes;
            int minHoughPints; 
            int thetaRes;
            float maxDistancePtsLine;
            int topLineRetries;
            int minSharedPoints;
            bool isExecuting;
            Mat points1;
            Mat points2;
            int nPoints;
            
        public:
            SeparableFundamentalMatFindCommand(InputArray _points1, InputArray _points2, int _imSizeHOrg, int _imSizeWOrg,
                float _inlierRatio, int _inlierThreashold, float _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
                int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints);
            
            ~SeparableFundamentalMatFindCommand()
            {
                points1.release();
                points2.release();
            }

            Mat Execute();

            static vector<top_line> FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
                const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
                int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio);

            //int FindMat(Mat& _fmatrix );
        };

        
    }
}

#endif // !_OPENCV_SFM_FINDER_H_