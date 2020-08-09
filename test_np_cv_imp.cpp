    
#include "np_cv_imp.hpp"

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