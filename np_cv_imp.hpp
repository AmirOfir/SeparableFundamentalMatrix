// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_NP_CV_IMP_H_
#define _OPENCV_NP_CV_IMP_H_

#include <opencv2/opencv.hpp>
#include <opencv2\core\core_c.h>

namespace cv { namespace separableFundamentalMatrix
{
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
                    for (int k = 0; k < reduce_count; k++)
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
        mat = mat.reshape(1, (int)mat.total());

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

    vector<int> IndexWhereLowerThan(const vector<float> &vec, float maxValue);

    template<typename _Tp>
    vector<_Tp> intersect1d(const vector<_Tp> &a, const vector<_Tp> &b)
    {
        vector<_Tp> ret;

        // Copy
        vector<_Tp> acopy = a, bcopy = b;

        // Sort
        sort(acopy.begin(), acopy.end());
        sort(bcopy.begin(), bcopy.end());

        // intersect
        set_intersection(acopy.begin(), acopy.end(), bcopy.begin(), bcopy.end(), back_inserter(ret));

        return ret;
    }

    // Helper - Converts an input array to vector
    template<class T>
    std::vector<T>& getVec(InputArray _input) {
        std::vector<T> *input;
        if (_input.isVector()) {
            input = static_cast<std::vector<T>*>(_input.getObj());
        } else {
            size_t length = _input.total();
            if (_input.isContinuous()) {
                T* data = reinterpret_cast<T*>(_input.getMat().data);
                input = new std::vector<T>(data, data + length);
            }
            else {
                input = new std::vector<T>;
                Mat mat = _input.getMat();
                for (size_t i = 0; i < mat.rows; i++)
                {
                    input->insert(input->end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols*mat.channels());
                }
            }
        }
        return *input;
    }

    //template <typename _Tp, typename _Ix>
    //vector<_Tp> ByIndices(const vector<_Tp> &vec, const vector<_Ix> &indices)
    //{
    //    vector<_Tp> ret;
    //    for (size_t i = 0; i < indices.size(); ++i )
    //        ret.push_back(vec[indices[i]]);
    //    return ret;
    //}


    template <typename _Tp>
    vector<Point_<_Tp>> ByIndices(InputArray _input, const vector<int> &indices)
    {
        CV_Assert(_input.dims() == 2);
        Point_<_Tp>* data = reinterpret_cast<Point_<_Tp>*>(_input.getMat().data);

        Mat mat = _input.getMat();
        cout << mat.at<float>(0, 0) << "," << mat.at<float>(0, 1);

        //// To vector
        //Point2f *data = (Point2f *)mat.data;
        //int length = mat.total();
        //std::vector<Point2f> vec;
        //vec.assign(data, data + length);

        vector<Point_<_Tp>> ret;
        for (auto index : indices)
            ret.push_back(data[index]);
        return ret;
    }

    template <typename _Tp>
    vector<Point_<_Tp>> projectPointsOnLine(const Point_<_Tp> &pt1, const Point_<_Tp> &pt2, InputArray _points)
    {
        vector<Point_<_Tp>> projections;
        vector<Point_<_Tp>> points = _points.getMat();
        _Tp dx = pt2.x - pt1.x, dy = pt2.y - pt1.y;
        _Tp d = dx * dx + dy * dy;
        if (d == 0) return projections;
        
        for (auto point : points)
        {
            _Tp a = (dy*(point.y - pt1.y) + dx * (point.x - pt1.x)) / d;
            projections.push_back(Point_<_Tp>(pt1.x + a * dx, pt1.y + a * dy));
        }

        return projections;
    }
}}


#endif // !_OPENCV_NP_CV_IMP_H_
