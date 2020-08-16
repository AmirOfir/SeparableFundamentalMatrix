#include "np_cv_imp.hpp"
#include <random>
#include <time.h>

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;


class TestGen
{
    /*default_random_engine random_engine_;
    normal_distribution<int> distribution_;*/
public:
    void genIntersection(vector<int> &a, vector<int> &b, vector<int> &intersection)
    {
        srand(time(0));

        set<int> numbers;
        while (numbers.size() < 15)
        {
            int n = rand() % 50;
            numbers.insert(n);
        }
        vector<int> shuffled(numbers.begin(), numbers.end());
        std::random_shuffle(shuffled.begin(), shuffled.end());

        for (size_t i = 0; i < 5; i++)
        {
            intersection.push_back(shuffled[i]);
            a.push_back(shuffled[i]);
            b.push_back(shuffled[i]);
        }
        for (size_t i = 5; i < 10; i++)
        {
            a.push_back(shuffled[i]);
        }
        for (size_t i = 10; i < 15; i++)
        {
            b.push_back(shuffled[i]);
        }

        std::random_shuffle(a.begin(), a.end());
        std::random_shuffle(b.begin(), b.end());
        std::sort(intersection.begin(), intersection.end());
    }
};

vector<int> MatToVec(Mat mat)
{
    std::vector<int> array;
    if (mat.isContinuous()) {
      // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
      array.assign((int*)mat.data, (int*)mat.data + mat.total()*mat.channels());
    } else {
      for (int i = 0; i < mat.rows; ++i) {
        array.insert(array.end(), mat.ptr<int>(i), mat.ptr<int>(i)+mat.cols*mat.channels());
      }
    }
    return array;
}

void test_reduceSum3d()
{
    const int sizes[] = { 10,10, 10 };
    Mat h(3, sizes, CV_8U);
    uchar *d = h.data;
    
    h.at<uchar>(0, 0, 0) = 2;
    h.at<uchar>(0, 0, 1) = 2;
    h.at<uchar>(0, 0, 2) = 2;
    h.at<uchar>(0, 0, 3) = 2;
    h.at<uchar>(0, 0, 4) = 2;
}

void test_intersect1d()
{
    TestGen testGen;
    vector<int> a, b, expected, result;
    testGen.genIntersection(a, b, expected);
    intersect1d(a.begin(), a.end(), b.begin(), b.end(), back_inserter(result));
    CV_Assert(expected.size() == result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        CV_Assert(expected[i] == result[i]);
    }
}

int __stdcall testMain()
{
    return 0;
}