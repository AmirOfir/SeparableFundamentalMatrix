#include "np_cv_imp.hpp"

using namespace cv;
using namespace std;
namespace cv {
    namespace separableFundamentalMatrix {
        Mat randomIntMat(int rows, int cols, int min, int max)
        {
            Mat ret(rows, cols, CV_32S);
            cv::randu(ret, Scalar(min), Scalar(max));

            return ret;
        }

    }
}