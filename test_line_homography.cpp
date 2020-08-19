#include "line_homography.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

void assert_same(Mat a, Mat b)
{
    double minval, maxval;
    Mat diff = (a - b);
    diff = cv::abs(diff);
    cv::minMaxLoc(diff, &minval, &maxval);
    bool areIdentical = minval < 0.001;
    assert(areIdentical);
}

void test_lineHomography()
{
    float data[] = { 0.59337338,  0.22072658, -0.12062497,  0.76461587 };
    Mat expected(2, 2, CV_32F, data);
    cout << expected.at<float>(0, 0) << "," << expected.at<float>(0, 1) << "," << expected.at<float>(1, 0) << "," << expected.at<float>(1, 1) << endl;
    
    vector<Vec4f> points
    {
       Vec4f(1404.90829526,  697.53539   ,  962.34180838,  354.54451947),
       Vec4f(1368.16594788,  663.27259867,  934.69049355,  333.70775526),
       Vec4f(1319.22244357,  617.63204506,  896.34750509,  304.81423557)
    };
    

    Mat ret = findLineHomography( points );   
}

void test_lineHomographyError()
{
    vector<Vec4f> data {
       Vec4f(1368.16594788,  663.27259867,  934.69049355,  333.70775526),
       Vec4f(1272.73827193,  574.28485614,  862.43231627,  279.25730285),
       Vec4f(1319.22244357,  617.63204506,  896.34750509,  304.81423557),
       Vec4f(1300.41905565,  600.09760313,  883.07281945,  294.81104056),
       Vec4f(1322.13497292,  620.34802247,  898.83713967,  306.69031014),
       Vec4f(1350.66748904,  646.95502272,  920.55055581,  323.0525459 ),
       Vec4f(1257.33395268,  559.92009684,  849.99620546,  269.88601942),
       Vec4f(1404.90829526,  697.53539   ,  962.34180838,  354.54451947),
       Vec4f(1291.72343069,  591.98880212,  876.50288223,  289.86023682),
       Vec4f(1316.54069944,  615.13127834,  894.4124638 ,  303.3560771 ),
       Vec4f(1308.0230798 ,  607.18846997,  888.29166853,  298.74372616),
       Vec4f(1209.51181494,  515.32523442,  814.3592433 ,  243.03163719),
       Vec4f(1247.61317678,  550.85532717,  842.31803851,  264.10010453)
    };

    double modelData[] = { 0.07462337,  0.71457505,
       -0.17591846,  0.6729536 };
    Mat model(2, 2, CV_64F, modelData);

    Mat result = lineHomographyError(model, data);

    double expectedData[] = {
        5.02337176e-10, 7.74789809e+00, 5.34901733e-10, 3.44015090e+00,
       2.72624837e-01, 1.06083669e+00, 9.02999960e+00, 4.82430977e-10,
       4.48391373e+00, 3.79033426e-01, 1.70863860e+00, 2.22412839e+01,
       1.03963197e+01
    };
    Mat expected(13, 1, CV_64F, expectedData);

    assert_same(result, expected);
}

void test_normalizeCoordinatesByLastCol()
{
    double data[] = { 2219.94646782, 2164.71005817, 2190.63588876, 2180.48830407,
        2192.53904327, 2209.13745737, 2155.20350628, 2241.08399732,
        2175.46603867, 2189.15668271, 2184.47775546, 2127.96149832,
        2149.33407543,
       1076.20692132,  980.85479499, 1025.60938874, 1008.09207029,
        1028.89471865, 1057.54781337,  964.44406262, 1112.69568652,
         999.42235971, 1023.05590195, 1014.97888055,  917.41741117,
         954.31192848 };
    Mat a(2,13,CV_64F, data);
    auto t = a.t();

    Mat normalized;
    normalizeCoordinatesByLastCol(t, normalized);

    double expectedData[] = {
       2.0627506 , 1.,
       2.20696281, 1.,
       2.13593588, 1.,
       2.16298528, 1.,
       2.1309654 , 1.,
       2.08892442, 1.,
       2.2346589 , 1.,
       2.01410325, 1.,
       2.1767234 , 1.,
       2.13982118, 1.,
       2.15223962, 1.,
       2.31951288, 1.,
       2.25223432, 1. };
    Mat expected(13, 2, CV_64F, expectedData);
    cout << normalized ;

    double minval, maxval;
    Mat diff = (expected - normalized);
    cv::minMaxLoc(diff, &minval, &maxval);
    bool areIdentical = minval < 0.001;
    assert(areIdentical);
}