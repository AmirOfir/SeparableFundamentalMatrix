#include "pch.h"
#include "..\SFM_finder.hpp"

using namespace cv;
using namespace cv::separableFundamentalMatrix;


template <typename _Tp>
void assert_same(Mat a, Mat b)
{
    Mat diff = (a - b);
    diff = cv::abs(diff);
    for (int row = 0; row < diff.rows; row++)
    {
        for (int col = 0; col < diff.cols; col++)
        {
            bool areIdentical = b.at<_Tp>(row,col) == a.at<_Tp>(row,col) ||
                diff.at<_Tp>(row, col) < 0.001;
            if (!areIdentical)
            {
                cout << "first is " << a << endl;
                cout << "second is " << b << endl;
            }
            assert(areIdentical);
        }
    }
}

TEST(TestSFMEstimatorCallback, CalculatesFundamentalMatrixPureOpenCV2d)
{
    double data[] = { -5.42591269e-06,  3.79209754e-05, -9.86489188e-02,
       -2.99228592e-05, -5.35287692e-06,  8.16081142e-02,
        7.15609947e-02, -5.44480665e-02,  1.00000000e+00 };
    Mat expected(3, 3, CV_64F, data);

    vector<Point2d> pL = {
       Point2d(4.68506371e+02, 7.44781247e+02),
       Point2d(1.24814415e+03, 5.50395695e+02),
       Point2d(6.72998873e+02, 6.93795535e+02),
       Point2d(1.29099915e+03, 5.92765503e+02),
       Point2d(1.10701306e+03, 1.07263513e+03),
       Point2d(1.27033679e+03, 6.02361389e+02),
       Point2d(2.09753098e+02, 3.64546967e+02),
       Point2d(5.86394897e+02, 7.86725952e+02)
    };
    vector<Point2d> pR = {
       Point2d(232.07621361, 360.12905516),
       Point2d(843.18101873, 263.33954774),
       Point2d(403.06458351, 333.04715373),
       Point2d(875.47491455, 291.22439575),
       Point2d(714.99407959, 547.44390869),
       Point2d(861.9050293 , 298.64364624),
       Point2d( 99.43354034, 158.61286926),
       Point2d(323.4100647 , 386.88360596)
    };
    Mat p3L(pL), p3R(pR);
    Mat m = cv::findFundamentalMat(p3L, p3R, FM_8POINT);

    assert_same<double>(m, expected);

}

TEST(TestSFMEstimatorCallback, CalculatesFundamentalMatrixPureOpenCV3d)
{
    double data[] = { -5.42591269e-06,  3.79209754e-05, -9.86489188e-02,
       -2.99228592e-05, -5.35287692e-06,  8.16081142e-02,
        7.15609947e-02, -5.44480665e-02,  1.00000000e+00 };
    Mat expected(3, 3, CV_64F, data);

    vector<Point3d> pL = {
       Point3d(4.68506371e+02, 7.44781247e+02, 1.00000000e+00),
       Point3d(1.24814415e+03, 5.50395695e+02, 1.00000000e+00),
       Point3d(6.72998873e+02, 6.93795535e+02, 1.00000000e+00),
       Point3d(1.29099915e+03, 5.92765503e+02, 1.00000000e+00),
       Point3d(1.10701306e+03, 1.07263513e+03, 1.00000000e+00),
       Point3d(1.27033679e+03, 6.02361389e+02, 1.00000000e+00),
       Point3d(2.09753098e+02, 3.64546967e+02, 1.00000000e+00),
       Point3d(5.86394897e+02, 7.86725952e+02, 1.00000000e+00)
    };
    vector<Point3d> pR = {
       Point3d(232.07621361, 360.12905516,   1.        ),
       Point3d(843.18101873, 263.33954774,   1.        ),
       Point3d(403.06458351, 333.04715373,   1.        ),
       Point3d(875.47491455, 291.22439575,   1.        ),
       Point3d(714.99407959, 547.44390869,   1.        ),
       Point3d(861.9050293 , 298.64364624,   1.        ),
       Point3d( 99.43354034, 158.61286926,   1.        ),
       Point3d(323.4100647 , 386.88360596,   1.        )
    };
    Mat p3L(pL), p3R(pR);
    Mat m = cv::findFundamentalMat(p3L, p3R, FM_8POINT);

    assert_same<double>(m, expected);

}

TEST(TestSFMEstimatorCallback, CalculatesFundamentalMatrix)
{
    double data[] = { 
        -3.84625600e-07,  2.17855930e-05, -2.78007384e-02,
        -2.43826609e-05, -9.14132784e-07,  3.54541238e-02,
         1.83089137e-02, -2.05466468e-02,  1.00000000e+00 };
    Mat expected(3, 3, CV_64F, data);

    vector<Point3d> pL = {
        Point3d(1.28279736e+03, 5.90549316e+02, 1),
        Point3d(1.15816443e+03, 1.11479822e+03, 1),
        Point3d(1.36326611e+03, 4.03504852e+02, 1),
        Point3d(1.15782727e+03, 1.15636804e+03, 1),
        Point3d(1.22063855e+03, 6.26720032e+02, 1),
        Point3d(4.68506371e+02, 7.44781247e+02, 1),
        Point3d(1.24814415e+03, 5.50395695e+02, 1),
        Point3d(6.72998873e+02, 6.93795535e+02, 1)
    };
    vector<Point3d> pR = {
       Point3d(8.69601440e+02, 2.90044403e+02, 1),
       Point3d(7.58411438e+02, 5.76697754e+02, 1),
       Point3d(9.25457458e+02, 2.23250793e+02, 1),
       Point3d(7.53959717e+02, 6.02285767e+02, 1),
       Point3d(8.27935425e+02, 3.16720428e+02, 1),
       Point3d(2.32076214e+02, 3.60129055e+02, 1),
       Point3d(8.43181019e+02, 2.63339548e+02, 1),
       Point3d(4.03064584e+02, 3.33047154e+02, 1)
    };
    Mat p3L(pL), p3R(pR);
    Mat m = cv::findFundamentalMat(p3L, p3R, FM_8POINT);

    assert_same<double>(m, expected);


    vector<Point2d> pointsL = {
     Point2d(1.28279736e+03, 5.90549316e+02  ),
     Point2d(1.15816443e+03, 1.11479822e+03  ),
     Point2d(1.36326611e+03, 4.03504852e+02  ),
     Point2d(1.15782727e+03, 1.15636804e+03  ),
     Point2d(1.22063855e+03, 6.26720032e+02)
    };
    vector<Point2d> pointsR = {
       Point2d(8.69601440e+02, 2.90044403e+02),
       Point2d(7.58411438e+02, 5.76697754e+02),
       Point2d(9.25457458e+02, 2.23250793e+02),
       Point2d(7.53959717e+02, 6.02285767e+02),
       Point2d(8.27935425e+02, 3.16720428e+02)
    };
    Mat m1(pointsL), m2(pointsR);

    vector<Point2d> fixedPointL = {
        Point2d(4.68506371e+02, 7.44781247e+02),
        Point2d(1.24814415e+03, 5.50395695e+02),
        Point2d(6.72998873e+02, 6.93795535e+02)
    };
    vector<Point2d> fixedPointR = {
        Point2d(2.32076214e+02, 3.60129055e+02),
        Point2d(8.43181019e+02, 2.63339548e+02),
        Point2d(4.03064584e+02, 3.33047154e+02)
    };
    Mat f1(fixedPointL), f2(fixedPointR);
    
    SFMEstimatorCallback sfm;
    sfm.setFixedMatrices(f1, f2);

    Mat model(3, 3, CV_64F);
    sfm.runKernel(m1, m2, model);

    

    assert_same<double>(model, expected);

}

TEST(TestSFMEstimatorCallback, CalculatesFundamentalMatrix2)
{
    vector<Point2d> pointsL = {
        Point2d(6.53061646e+02, 9.07054993e+02  ),
        Point2d(3.93640442e+02, 6.04780334e+02  ),
        Point2d(5.27386780e+02, 7.41797363e+02  ),
        Point2d(7.25550476e+02, 3.15516724e+02  ),
        Point2d(1.21212366e+03, 5.87335327e+02)
    };
    vector<Point2d> pointsR = {
        Point2d(3.73052551e+02, 4.70658569e+02),
        Point2d(1.90753128e+02, 2.76573181e+02),
        Point2d(2.77985168e+02, 3.56134644e+02),
        Point2d(9.10476990e+02, 1.95234711e+02),
        Point2d(8.20373901e+02, 2.91974823e+02)
    };
    Mat m1(pointsL), m2(pointsR);

    vector<Point2d> fixedPointL = {
        Point2d(4.47340762e+02, 7.41009944e+02),
        Point2d(1.27719872e+03, 7.99039284e+02),
        Point2d(6.94567659e+02, 7.58297739e+02)
    };
    vector<Point2d> fixedPointR = {
        Point2d(2.17158396e+02, 3.56653013e+02),
        Point2d(8.74205341e+02, 4.25711462e+02),
        Point2d(4.21864342e+02, 3.78168485e+02)
    };
    Mat f1(fixedPointL), f2(fixedPointR);
    
    SFMEstimatorCallback sfm;
    sfm.setFixedMatrices(f1, f2);

    Mat model;// (3, 3, CV_64F);
    sfm.runKernel(m1, m2, model);

    double data[] = { 
        -2.49008112e-06, -3.07230209e-05,  4.10604254e-03,
        4.47674168e-05, -1.10741817e-05,  1.35762501e-02,
       -4.99530539e-04, -7.29981879e-03,  1.00000000e+00 };
    Mat expected(3, 3, CV_64F, data);

    assert_same<double>(model, expected);
}

TEST(TestSFMEstimatorCallback, FindInliers)
{
    double modelData[] = { 
        -2.49008112e-06, -3.07230209e-05,  4.10604254e-03,
        4.47674168e-05, -1.10741817e-05,  1.35762501e-02,
       -4.99530539e-04, -7.29981879e-03,  1.00000000e+00 };
    Mat model(3, 3, CV_64F, modelData);

    vector<Point2d> pointsL = {
        Point2d(6.53061646e+02, 9.07054993e+02  ),
        Point2d(3.93640442e+02, 6.04780334e+02  ),
        Point2d(5.27386780e+02, 7.41797363e+02  ),
        Point2d(7.25550476e+02, 3.15516724e+02  ),
        Point2d(1.21212366e+03, 5.87335327e+02)
    };
    vector<Point2d> pointsR = {
        Point2d(3.73052551e+02, 4.70658569e+02),
        Point2d(1.90753128e+02, 2.76573181e+02),
        Point2d(2.77985168e+02, 3.56134644e+02),
        Point2d(9.10476990e+02, 1.95234711e+02),
        Point2d(8.20373901e+02, 2.91974823e+02)
    };
    Mat m1(pointsL), m2(pointsR);

    vector<Point2d> fixedPointL = {
        Point2d(4.47340762e+02, 7.41009944e+02),
        Point2d(1.27719872e+03, 7.99039284e+02),
        Point2d(6.94567659e+02, 7.58297739e+02)
    };
    vector<Point2d> fixedPointR = {
        Point2d(2.17158396e+02, 3.56653013e+02),
        Point2d(8.74205341e+02, 4.25711462e+02),
        Point2d(4.21864342e+02, 3.78168485e+02)
    };
    Mat f1(fixedPointL), f2(fixedPointR);

    SFMEstimatorCallback sfm;

    Mat err;
    sfm.computeError(m1, m2, model, err);

    float errExpectedData[] = { 0.10464236, 9.21043061, 2.76470398, 2.18282727, 2.22311052 };
    Mat expected(5, 1, CV_32F, errExpectedData);

    assert_same<float>(err, expected);
}