
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

//#include "opencv2/core/private.hpp"
//#include "opencv2/core/traits.hpp"
//#include "opencv2/core/types.hpp"
//#include "opencv2/core/mat.hpp"
////#include "opencvlight/types.hpp"
//#include "opencv2/core/core_c.h"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/calib3d.hpp"
#include "opencvlight/cv_cpu_dispatch.h"
#include "opencvlight/cvdef.h"
#include "opencvlight/traits.h"
#include "opencvlight/types.hpp"
#include "opencvlight/mat.hpp"
#include "opencvlight/inputarray.hpp"
#include <vector>
#include <numeric>

#define CV_Assert( expr ) do { if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ ); } while(0)

namespace cv 
{

#define CV_REDUCE_SUM 0
#define CV_REDUCE_AVG 1
#define CV_REDUCE_MAX 2
#define CV_REDUCE_MIN 3
void error( const exception& exc )
{

    throw exc;
}

void error(int _code, const string& _err, const char* _func, const char* _file, int _line)
{
    throw exception(_err.c_str());
}

}

namespace cv {
namespace separableFundamentalMatrix {

using namespace cv;

struct line_info
{
    std::vector<int> matching_indexes;
    Point3d line_eq_abc;
    Point3d line_eq_abc_norm;
    Point2d bottom_left_edge_point;
    Point2d top_right_edge_point;
    double max_distance;
    int line_index;
};


class top_line
{
public:
    bool empty() { return num_inliers == 0; }
    int num_inliers;
    std::vector<Point2d> line_points_1;
    std::vector<Point2d> line_points_2;
    int line1_index;
    int line2_index;
    std::vector<int> inlier_selected_index;
    std::vector<Point2d> selected_line_points1;
    std::vector<Point2d> selected_line_points2;
    double max_dist;
    double min_dist;
    double homg_err;
};

}    
}

#endif
