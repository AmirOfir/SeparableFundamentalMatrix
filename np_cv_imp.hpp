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
    
    template <typename _Tp>
    bool lexicographicalSort2d(Point_<_Tp> a, Point_<_Tp> b)
    {
        return std::tie(a.x, a.y) < std::tie(b.x, b.y);// < b.x || (a.x == b.x && a.y < b.y);
    }
    
    template <typename _Tp>
    bool lexicographicalSort3d(const Point3_<_Tp> &p1, const Point3_<_Tp> &p2)
    {
        return std::tie(p1.x, p1.y, p1.z) < std::tie(p2.x, p2.y, p2.z);
    }

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
    vector<Point3_<_Tp>> indices(InputArray _mat)
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

    template <class InputIterator, class UnaryPredicate>
    vector<int64> index_if(InputIterator first, InputIterator last, UnaryPredicate pred)
    {
        vector<int64> ret;

        int64 i = 0;
        while (first != last) 
        {
            if (pred(*first)) 
                ret.push_back(i);
            
            ++i;
            ++first;
        }
        return ret;
    }

    template <class _InIt1, class _InIt2, class _OutIt> inline
    void intersect1d(_InIt1 _First1,  _InIt1 _Last1, _InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
    {
        // Copy
        vector<typename iterator_traits<_InIt1>::value_type> acopy(_First1, _Last1);
        vector<typename iterator_traits<_InIt2>::value_type> bcopy(_First2, _Last2);

        // Sort
        std::sort(acopy.begin(), acopy.end());
        std::sort(bcopy.begin(), bcopy.end());

        // Intersect
        set_intersection(acopy.begin(), acopy.end(), bcopy.begin(), bcopy.end(), _Dest);
    }

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

    template <typename _Tp>
    vector<Point_<_Tp>> byIndices(InputArray _input, const vector<int> &indices)
    {
        CV_Assert(_input.dims() == 2);
        Mat mat = _input.isMat() ? _input.getMat() : _input.getMat().t();
        
        vector<Point_<_Tp>> ret;
        for (auto index : indices)
            ret.push_back(Point_<_Tp>(mat.at<float>(index, 0), mat.at<float>(index, 1)));
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

    template <typename _Tp>
    vector<size_t> index_unique(const vector<Point_<_Tp>> &_points)
    {
        typedef tuple<Point_<_Tp>, size_t> point_index;
        auto sort_point_index = [](point_index a, point_index b)->bool { return lexicographicalSort2d(std::get<0>(a), std::get<0>(b)); };
        auto compare_point_index = [](point_index a, point_index b)->bool {return std::get<0>(a) == std::get<0>(b); };

        // Add the index field
        vector<point_index> vec;
        for (size_t i = 0; i < _points.size(); ++i)
            vec.push_back(point_index(_points[i], i));
        
        // Sort
        std::sort(vec.begin(), vec.end(), sort_point_index);

        // Unique
        vec.erase( std::unique(vec.begin(), vec.end(), compare_point_index), vec.end() ); 
        
        // Get the index field
        vector<size_t> ret;
        for (auto p : vec)
            ret.push_back(get<1>(p));
        
        return ret;
    }

    Mat randomIntMat(int rows, int cols, int min, int max);

    template <typename _Tp>
    Mat PointVectorToMat(const vector<Point_<_Tp>> &vec)
    {
        Mat mat((int)vec.size(), 2, traits::Type<_Tp>::value, (void*)vec.data());
        return mat;
    }

    /*
    template <typename _Tp>
    vector<Vec<_Tp,4>> concatenate(const vector<Point_<_Tp>> &points1, const vector<Point_<_Tp>> &points2)
    {
        vector<Vec<_Tp,4>> points;
        for (size_t i = 0; i < points1.size(); i++)
        {
            points.push_back(Vec<_Tp,4>(points1[i].x,points1[i].y, points2[i].x,points2[i].y));
        }
        return points;
    }
    */
    /*
    template <typename _Tp>
    vector<vector<Vec<_Tp,4>>> randomSamples(int numIterations, int k, const vector<Vec<_Tp,4>> &points)
    {
        Mat randomIndices = randomIntMat(numIterations, k, 0, points.size());
        randomIndices.begin
        vector<vector<Vec<_Tp,4>>> ret;

        for (size_t iteration = 0; iteration < numIterations; iteration++)
        {
            vector<Vec<_Tp,4>> curr;
            for (int kIx = 0; kIx < k; kIx++)
            {
                curr.push_back(points[randomIndices.at<int>(iteration, kIx)]);
            }
            ret.push_back(curr);
        }

        return ret;
    }
    */

    void matrixVectorElementwiseMultiplication(InputArray _matrix, InputArray _vector, OutputArray _ret);
    
}}


#endif // !_OPENCV_NP_CV_IMP_H_
