#include "line_geometry.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

void test_intervalEndpoints()
{
    vector<Point2d> vec
    {
       Point2d(1308.0230798 ,  607.18846997),
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1257.33395268,  559.92009684),
       Point2d(1404.90829526,  697.53539),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1209.51181494,  515.32523442)
    };

    auto result = intervalEndpoints(vec);
    CV_Assert(result.distance - 267.170966 < 1e-3);
    CV_Assert( (result.firstIdx == 4 && result.secondIdx == 6) ||
               (result.firstIdx == 6 && result.secondIdx == 4) );
}

void test_intervalMedian()
{
    vector<Point2d> vec
    {
       Point2d(1308.0230798 ,  607.18846997),
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1257.33395268,  559.92009684),
       Point2d(1404.90829526,  697.53539),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1209.51181494,  515.32523442)
    };

    auto result = intervalMedian(vec, 4, 6);
    CV_Assert(result.minDistance - 132.47381225 < 0.0001);
}