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

        void matrixVectorElementwiseMultiplication(InputArray _matrix, InputArray _vector, OutputArray _ret)
        {
            Mat matrix = _matrix.getMat();
            Mat vector = _vector.getMat();
            
            CV_Assert(matrix.type() == vector.type() && matrix.rows == vector.rows && vector.cols == 1);
            vector = Scalar(1) / vector;
            
            Mat ret = matrix.col(0) / vector;
            for (size_t i = 1; i < matrix.cols; i++)
            {
                cv::hconcat(ret, matrix.col(i) / vector, ret);
            }

            _ret.assign(ret);
        }

    }
}