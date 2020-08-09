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
#ifndef _OPENCV_NP_CV_IMP_H_
#define _OPENCV_NP_CV_IMP_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2\core\core_c.h>
using namespace cv;
using namespace std;

template <typename TSource, typename TDest>
void reduce3d(InputArray _src, OutputArray _dst, int dim, int rtype, int dtype)
{
    CV_Assert( _src.dims() == 3 );
    CV_Assert( dim <= 2 && dim >= 0);
    Mat src = _src.getMat();

    // Create dst
    int sizes[2];
    if (dim == 0)
    {
        sizes[0] = src.size[1];
        sizes[1] = src.size[2];
    }
    else if (dim == 1)
    {
        sizes[0] = src.size[0];
        sizes[1] = src.size[2];
    }
    else
    {
        sizes[0] = src.size[0];
        sizes[1] = src.size[1];
    };
    _dst.create(2, sizes, dtype);
    Mat dst = _dst.getMat();
    
    // Fill
    int reduce_count = src.size[dim];
    parallel_for_(Range(0, sizes[0]), [&](const Range& range) {

            for (int i = range.start; i < range.end; i++)
            {
                for (int j = 0; j < sizes[1]; j++)
                {
                    TDest c = 0;
                    for (int k = 0; k < reduce_count; k++)
                    {
                        TSource s = src.at<TSource>(dim == 0 ? k : i, dim == 1 ? k : j, dim == 2 ? k : j);
                        if (rtype == CV_REDUCE_SUM || rtype == CV_REDUCE_AVG)
                            c += s;
                        else if (rtype == CV_REDUCE_MAX)
                            c = max(s, c);
                        else if (rtype == CV_REDUCE_MIN)
                            c = min(s, c);
                    }
                    if (rtype == CV_REDUCE_AVG)
                        c = c / reduce_count;
                    dst.at<TDest>(i, j) = c;
                }
            }
        });
}

template <typename TSource, typename TDest>
void reduceSum3d(InputArray _src, OutputArray _dst, int dtype)
{
    CV_Assert( _src.dims() == 3 );
    Mat src = _src.getMat();

    // Create dst
    int sizes[]{ src.size[0], src.size[1] };
    _dst.create(2, sizes, dtype);
    Mat dst = _dst.getMat();
    
    // Fill
    int reduce_count = src.size[2];
    parallel_for_(Range(0, sizes[0]), [&](const Range& range) {
        
        for (int i = range.start; i < range.end; i++)
        {
            for (int j = 0; j < sizes[1]; j++)
            {
                TDest c = 0;
                for (size_t k = 0; k < reduce_count; k++)
                {
                    c += src.at<TSource>( i, j, k );
                }
                dst.at<TDest>(i, j) = c;
            }
        }
        });
}

template<typename _Tp>
vector<Point3_<_Tp>> Indices(InputArray _mat)
{
    CV_Assert(_mat.dims() == 2);
    Mat mat = _mat.getMat();
    
    const int cols = mat.cols;

    // Flatten
    mat = mat.reshape(1, mat.total());

    // Create return value
    vector<Point3_<_Tp>> ret;
    ret.reserve(mat.total());

    _Tp *arr = (_Tp*)mat.data;
    int currRow = 0;
    int currCol = 0;
    for (auto v_ix = 0; v_ix < mat.total(); v_ix++)
    {
        ret.push_back(Point3_<_Tp>( arr[v_ix], currRow, currCol));
        
        ++currCol;
        if (currCol == cols)
        {
            ++currRow;
            currCol = 0;
        }
    }
    return ret;
}

vector<int> IndexWhereLowerThan(const vector<float> &vec, float maxValue)
{
    vector<int> ret;
    for (auto i = 0; i < vec.size(); i++)
    {
        if (vec[i] < maxValue)
            ret.push_back(i);
    }
    
    return ret;
}
#endif // !_OPENCV_NP_CV_IMP_H_
