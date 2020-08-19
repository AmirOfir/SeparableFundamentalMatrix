// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_MATCHING_POINTS_H_
#define _OPENCV_MATCHING_POINTS_H_

#include <opencv2/opencv.hpp>
#include <opencv2\core\core_c.h>
#include "np_cv_imp.hpp"

namespace cv { namespace separableFundamentalMatrix
{
    using namespace cv;
    using namespace std;
    
    template <typename _Tp>
    class MatchingPoints
    {
    public:
        Point_<_Tp> left;
        Point_<_Tp> right;

        static vector<MatchingPoints> FromVectors(const vector<Point_<_Tp>> &left, const vector<Point_<_Tp>> &right)
        {
            CV_Assert(left.size() == right.size());
            vector<MatchingPoints> ret;
            for (size_t i = 0; i < left.size(); i++)
            {
                MatchingPoints curr;
                curr.left = left[i];
                curr.right = right[i];
                ret.push_back(curr);
            }
            return ret;
        }

        static vector<vector<MatchingPoints>> RandomSamples(const vector<MatchingPoints> &source, uint iterations, uint sizeOfSample)
        {
            vector<vector<MatchingPoints>> ret;
            Mat randomIndices = randomIntMat(iterations, sizeOfSample, 0, source.size());
            for (uint iteration = 0; iteration < iterations; iteration++)
            {
                vector<MatchingPoints> curr;
                ret.push_back(curr);
                for (uint i = 0; i < sizeOfSample; i++)
                {
                    curr.push_back(source[randomIndices.at<int>(iteration, i)]);
                }
            }
            return ret;
        }

        static void ToMatrices(const vector<MatchingPoints> &source, OutputArray _left, OutputArray _right)
        {
            CV_Assert(source.size());
            vector<Point_<_Tp>> leftVec, rightVec;
            for (size_t i = 0; i < source.size(); i++)
            {
                leftVec.push_back(source[i].left);
                rightVec.push_back(source[i].right);
            }
            Mat leftMat = PointVectorToMat(leftVec);
            Mat rightMat = PointVectorToMat(rightVec);
            cout << leftMat;
            _left..assign(leftMat);
            _right.assign(rightMat);
        }
    };
}}


#endif // !_OPENCV_MATCHING_POINTS_H_
