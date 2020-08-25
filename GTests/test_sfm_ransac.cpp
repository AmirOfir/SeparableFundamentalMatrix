#include "pch.h"
#include "..\sfm_ransac.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

TEST(SFMRansac, TestHCoordinates) {
    Mat1d data(2,4);
    for (size_t row = 0; row < 2; row++)
    {
        for (size_t col = 0; col < data.cols; col++)
        {
            data(row, col) = row * 3 + col;
        }
    }
    h_coordinates(data);

    EXPECT_EQ(data.rows, 3);
    EXPECT_EQ(data.cols, 4);
    
    for (size_t row = 0; row < 2; row++)
    {
        for (size_t col = 0; col < data.cols; col++)
        {
            EXPECT_EQ(data(row, col), row * 3 + col);
        }
    }

    for (size_t col = 0; col < data.cols; col++)
    {
        EXPECT_EQ(data(2, col), 1);
    }
}

TEST(SFMRansac, TestPrepareDataForRansac)
{
    Mat1d data1(2,4);
    Mat1d data2(2,4);
    for (size_t row = 0; row < 2; row++)
    {
        for (size_t col = 0; col < 4; col++)
        {
            data1(row, col) = row * 3 + col;
            data1(row, col) = row * 3 + col + 1;
        }
    }
    Mat data = prepareDataForRansac(data1.t(), data2.t()).t();

    EXPECT_EQ(data.rows, 6);
    EXPECT_EQ(data.cols, 4);
    
    for (size_t row = 0; row < 2; row++)
    {
        for (size_t col = 0; col < data.cols; col++)
        {
            EXPECT_EQ(data.at<double>(row, col), data1(row, col));
            EXPECT_EQ(data.at<double>(row+3, col), data2(row, col));
        }
    }

    for (size_t col = 0; col < data.cols; col++)
    {
        EXPECT_EQ(data.at<double>(2, col), 1);
        EXPECT_EQ(data.at<double>(5, col), 1);
    }
}

TEST(SFMRansac, TestPrepareLinesForRansac)
{
    top_line l1;
    l1.selected_line_points1.push_back(Point2f(468.50637119, 744.78124722));
    l1.selected_line_points1.push_back(Point2f(1248.14415368, 550.39569517));
    l1.selected_line_points1.push_back(Point2f(672.99887258, 693.7955348));
    l1.selected_line_points2.push_back(Point2f(232.07621361, 360.12905516));
    l1.selected_line_points2.push_back(Point2f(843.18101873, 263.33954774));
    l1.selected_line_points2.push_back(Point2f(403.06458351, 333.04715373));
    top_line l2;
    l2.selected_line_points1.push_back(Point2f(1051.10424571,  481.69255083));
    l2.selected_line_points1.push_back(Point2f(1333.12474263,  625.3891296 ));
    l2.selected_line_points1.push_back(Point2f(1222.70688883,  569.12843933));
    l2.selected_line_points2.push_back(Point2f(707.48830855, 233.018089));
    l2.selected_line_points2.push_back(Point2f(906.30520044, 309.33674094));
    l2.selected_line_points2.push_back(Point2f(826.96604438, 278.88129332));

    auto actual = prepareLinesForRansac({ l1,l2 });

    for (size_t row = 0; row < 3; row++)
    {
        ASSERT_EQ(actual[0].at<float>(row, 0), l1.selected_line_points1[row].x);
        ASSERT_EQ(actual[0].at<float>(row, 1), l1.selected_line_points1[row].y);
        ASSERT_EQ(actual[0].at<float>(row, 2), 1);
        ASSERT_EQ(actual[0].at<float>(row, 3), l1.selected_line_points2[row].x);
        ASSERT_EQ(actual[0].at<float>(row, 4), l1.selected_line_points2[row].y);
        ASSERT_EQ(actual[0].at<float>(row, 5), 1);

        EXPECT_EQ(actual[1].at<float>(row, 0), l2.selected_line_points1[row].x);
        EXPECT_EQ(actual[1].at<float>(row, 1), l2.selected_line_points1[row].y);
        EXPECT_EQ(actual[1].at<float>(row, 2), 1);
        EXPECT_EQ(actual[1].at<float>(row, 3), l2.selected_line_points2[row].x);
        EXPECT_EQ(actual[1].at<float>(row, 4), l2.selected_line_points2[row].y);
        EXPECT_EQ(actual[1].at<float>(row, 5), 1);
    }
}
