// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "sfm_ransac.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

namespace cv {
    namespace separableFundamentalMatrix {

        void h_coordinates(InputOutputArray _pts)
        {
            Mat pts = _pts.getMat();
            Mat zeros = Mat::ones(1, pts.cols, pts.type());
            vconcat(pts, zeros, pts);
            _pts.assign(pts);
        }

        Mat prepareDataForRansac(InputArray _pts1, InputArray _pts2)
        {
            Mat pts1 = _pts1.getMat();
            Mat pts2 = _pts2.getMat();

            Mat x1n = pts1.t();
            h_coordinates(x1n);

            Mat x2n = pts2.t();
            h_coordinates(x2n);

            Mat data;
            cv::vconcat(x1n, x2n, data);
            data = data.t();
            return data;
        }

        vector<Mat> prepareLinesForRansac(const vector<top_line> &topMatchingLines)
        {
            vector<Mat> arrlines;

            for (auto topLine : topMatchingLines)
            {
                CV_Assert(topLine.selected_line_points1.size() && topLine.selected_line_points2.size());

                // Selected  points on the line
                Mat line_x1n = pointVectorToMat(topLine.selected_line_points1);
                line_x1n = line_x1n.t();
                h_coordinates(line_x1n);
        
                Mat line_x2n = pointVectorToMat(topLine.selected_line_points2);
                line_x2n = line_x2n.t();
                h_coordinates(line_x2n);

                Mat line;
                cv::hconcat(line_x1n, line_x2n, line);

                arrlines.push_back(line);
            }
            return arrlines;
        }

        
        FastFundamentalMatrixRansacResult fastFundamentalMatrixRansac(uint maxNumIterations, Mat data, 
            int inlier_threshold, const vector<Mat> &arrlines)
        {
            FastFundamentalMatrixRansacResult ret;
            return ret;
        }

    }
}