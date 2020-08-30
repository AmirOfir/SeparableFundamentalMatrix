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
    vector<int> genNumbers()
    {
        srand(time(0));

        set<int> numbers;
        while (numbers.size() < 15)
        {
            int n = rand() % 100;
            numbers.insert(n);
        }
        vector<int> shuffled(numbers.begin(), numbers.end());
        std::random_shuffle(shuffled.begin(), shuffled.end());
        return shuffled;
    }

    void genIntersection(vector<int> &a, vector<int> &b, vector<int> &intersection)
    {
        auto shuffled = genNumbers();

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
    Mat genMat()
    {
        srand(time(0));
        Mat ret(100 ,2, CV_32F);

        for (size_t i = 0; i < 100; i++)
        {
            ret.at<float>(i, 0) = rand();
            ret.at<float>(i, 1) = rand();
        }
        return ret;
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

void test_byIndices()
{
    TestGen testGen;
    Mat m = testGen.genMat();
    vector<int> indices = testGen.genNumbers();
    vector<Point2d> result = byIndices<double>(m, indices);
}

void test_matrixVectorElementwiseMultiplication()
{
    Mat1f mat(5,2);
    for (size_t row = 0; row < mat.rows; row++)
    {
        for (size_t col = 0; col < mat.cols; col++)
        {
            mat(row, col) = col;
        }
    }
    Mat mat1 = mat;

    Mat1f vec(5, 1);
    for (size_t row = 0; row < vec.rows; row++)
    {
        vec(row, 0) = row + 1;
    }
    Mat vec1 = vec;
    Mat m = matrixVectorElementwiseMultiplication<float>(mat, vec);


}