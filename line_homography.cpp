#include "line_homography.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;
namespace cv {
    namespace separableFundamentalMatrix {

void normalizeCoordinatesByLastCol(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    Mat lastCol = Scalar(1.) / (src.col(src.cols - 1) + 1e-10);
    matrixVectorElementwiseMultiplication(_src, lastCol, _dst);
/*def normalize_coordinates(pts):
    return pts / (pts[:, -1].reshape((-1, 1))+1e-10)#avoid divid by zero*/
}

void FlattenToMat(const vector<Vec4f> &data, OutputArray _dst)
{
    CV_Assert(data.size());
    Mat mat(data[0]);
    for (size_t i = 1; i < data.size(); i++)
    {
        mat.push_back(data[i]);
    }
    _dst.assign(mat);
}






//vector<MatchingPoints> randomSamples(int iterations, int sizeOfSample, const MatchingPoints &matchingPoints)
//{
//    vector<MatchingPoints> ret;
//    Mat randomIndices = randomIntMat(iterations, sizeOfSample, 0, matchingPoints.left.size());
//    for (int iteration = 0; iteration < iterations; iteration++)
//    {
//
//    }
//}




    }
}