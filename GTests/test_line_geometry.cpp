#include "pch.h"
#include "..\line_geometry.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;


TEST(LineGeometry, TestIntervalEndpoints) {
  vector<Point2f> vec
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
    EXPECT_TRUE(abs(result.distance - 267.170966) < 1e-3);
    EXPECT_TRUE( (result.firstIdx == 4 && result.secondIdx == 6) ||
               (result.firstIdx == 6 && result.secondIdx == 4) );
}

TEST(LineGeometry, TestIntervalEndpoints2) {
  vector<Point2f> vec
    {
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1257.33395268,  559.92009684),
       Point2d(1404.90829526,  697.53539   ),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1308.0230798 ,  607.18846997),
       Point2d(1209.51181494,  515.32523442)
    };

    auto result = intervalEndpoints(vec);
    EXPECT_TRUE(abs(result.distance - 267.1709664598408) < 1e-3);
    EXPECT_TRUE( (result.firstIdx == 3 && result.secondIdx == 6) ||
               (result.firstIdx == 6 && result.secondIdx == 3) );
}

TEST(LineGeometry, TestIntervalEndpoints3) {
    vector<Point2f> vec
    {
       Point2d(1368.16594788,  663.27259867),
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1291.72343069,  591.98880212),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1308.0230798 ,  607.18846997)
    };

    auto result = intervalEndpoints(vec);
    EXPECT_TRUE(abs(result.distance - 104.52195025372666) < 1e-3);
    EXPECT_TRUE(result.firstIdx == 0 || result.firstIdx == 3 || result.firstIdx == 1);
    EXPECT_TRUE(result.secondIdx == 0 || result.secondIdx == 3 || result.secondIdx == 1);
        
        
}

/*
TEST(LineGeometry, TestIntervalEndpoints4) {
  vector<Point2f> vec
    {
       Point2d(1319.22244357,  617.63204506),
       Point2d(1300.41905565,  600.09760313),
       Point2d(1257.33395268,  559.92009684),
       Point2d(1291.72343069,  591.98880212),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1308.0230798 ,  607.18846997),
       Point2d(1247.61317678,  550.85532717)
    };

    auto result = intervalEndpoints(vec);
    EXPECT_TRUE(result.distance - 97.91331442090005 < 1e-3);
    EXPECT_TRUE( (result.firstIdx == 3 && result.secondIdx == 0) ||
               (result.firstIdx == 0 && result.secondIdx == 3) );
}
*/

TEST(LineGeometry, TestIntervalPointClosestToCenter) {
    vector<Point2f> vec
    {
       Point2d(1308.0230798 ,  607.18846997),
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1257.33395268,  559.92009684),
       Point2d(1404.90829526,  697.53539),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1209.51181494,  515.32523442)
    };

    auto result = intervalPointClosestToCenter(vec, 4, 6);
    CV_Assert(abs(result.minDistance - 132.47381225) < 0.0001);
}

TEST(LineGeometry, TestIntervalPointClosestToCenter1) {
    vector<Point2f> vec
    {
       Point2d(1368.16594788,  663.27259867),
       Point2d(1322.13497292,  620.34802247),
       Point2d(1350.66748904,  646.95502272),
       Point2d(1291.72343069,  591.98880212),
       Point2d(1316.54069944,  615.13127834),
       Point2d(1308.0230798 ,  607.18846997)
    };

    auto result = intervalPointClosestToCenter(vec, 0, 3);
    EXPECT_EQ(result.midPointIdx, 1);
    CV_Assert(abs(result.minDistance - 41.58253575164936) < 0.001);
}