#include "line_homography.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;
namespace cv {
    namespace separableFundamentalMatrix {

/*
    Find 1D Homography

     input : kx4, k is at least 3
     output: The 2x2 (1d) line homography between the points
*/
Mat findLineHomography(const vector<Vec4f> &points)
{
    int numPoints = (int)points.size();
    Mat A = Mat::zeros(numPoints, 4, CV_32F);

    for (int n = 0; n < numPoints; n++)
    {
        A.at<float>(n, 0) = points[n][0] * points[n][3];
        A.at<float>(n, 1) = points[n][1] * points[n][3];
        A.at<float>(n, 2) = - points[n][0] * points[n][2];
        A.at<float>(n, 3) = - points[n][1] * points[n][2];
    }
    cv::SVD svd(A, SVD::Flags::FULL_UV);
    
    return svd.vt.row(3);
}

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



Mat lineHomographyError(Mat model, const vector<Vec4f> &data)
{
    Mat img(data);
    img = img.reshape(4, img.rows);
    cout << img;
    Mat src = img.colRange(0,1);
    Mat dst = img.colRange(2,3);
    try
    {
        auto dst_H = model * dst.t();
        auto src_H = model.inv() * dst.t();
        Mat resultL, resultR;

        normalizeCoordinatesByLastCol(src_H.t(), resultL);
        normalizeCoordinatesByLastCol(dst_H.t(), resultR);

        matrixVectorElementwiseMultiplication(resultL, src.col(src.cols - 1), resultL);
        matrixVectorElementwiseMultiplication(resultR, dst.col(dst.cols - 1), resultR);

        cv::pow(src - resultL, 2, resultL);
        cv::pow(src - resultR, 2, resultR);

        Mat result = resultL + resultR;
        cv::reduce(result, result, 1, CV_REDUCE_SUM, result.type());
        cv::sqrt(result, result);
        
        return result;
    }
    catch (...)
    {
        return Mat();
    }
}
/*
def homography_err(data,model):
    pts_src =data[:,0:2]
    pts_dest=data[:,2:4]
    pts_dest_H=np.dot(model,pts_src.T)
    try:
        pts_src_H = np.dot(np.linalg.inv(model), pts_dest.T)
        return np.sqrt(np.sum((  pts_src -normalize_coordinates(pts_src_H.T) *pts_src[:,-1].reshape((-1, 1))) **2+
                              (pts_dest  -normalize_coordinates(pts_dest_H.T)*pts_dest[:,-1].reshape((-1, 1)))**2,axis=1))
    except:
        return np.inf
        */

void cv::separableFundamentalMatrix::lineRansac(int numIterations, const vector<Point2f> &matchingPoints1, const vector<Point2f> &matchingPoints2, float inlierTh)
{
    CV_Assert(matchingPoints1.size() && matchingPoints1.size() == matchingPoints2.size());
    const int k = 3;
    auto data = concatenate(matchingPoints1, matchingPoints2);
    auto dataSamples = randomSamples(numIterations, k, data);
    vector<Mat> modelSamples;
    for (auto x : dataSamples)
        modelSamples.push_back(findLineHomography(x));
}
        /*def ransac_get_line_inliers(n_iters,line1_pts,line2_pts,inlier_th=0.35):
    data           = np.concatenate((line1_pts,line2_pts),axis=1)
    random_samples = [random.sample(list(np.arange(len(data))), k=3) for _ in range(n_iters)]
    data_samples   = [data[x, :] for x in random_samples]
    model_samples  = [line_homography(x) for x in data_samples]
    model_errs     = [homography_err(data,model_s) for model_s in model_samples]
    model_inliers  = [np.sum(model_err<inlier_th) for model_err in model_errs]
    best_idx_ransac= np.argmax(model_inliers)
    inliers_idx    = np.arange(len(data))[model_errs[best_idx_ransac]<inlier_th]
    return inliers_idx,np.mean(model_errs[best_idx_ransac][inliers_idx])*/


    }
}