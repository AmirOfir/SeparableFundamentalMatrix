// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_LINE_HOMOGRAPHY_H_
#define _OPENCV_LINE_HOMOGRAPHY_H_

#include <opencv2/opencv.hpp>
#include "np_cv_imp.hpp"
#include "matching_points.hpp"

namespace cv {
    namespace separableFundamentalMatrix {
        /*
            Find 1D Homography

             input : kx4, k is at least 3
             output: The 2x2 (1d) line homography between the points
        */
        template <typename _Tp>
        Mat findLineHomography(const VecMatchingPoints<_Tp> &matchingPoints)
        {
            int numPoints = (int)matchingPoints.size();
            Mat A = Mat::zeros(numPoints, 4, CV_32F);

            for (int n = 0; n < numPoints; n++)
            {
                MatchingPoints<_Tp> point = matchingPoints[n];
                
                A.at<float>(n, 0) = point.left.x * point.right.y;
                A.at<float>(n, 1) = point.left.y * point.right.y;
                A.at<float>(n, 2) = - point.left.x * point.right.x;
                A.at<float>(n, 3) = - point.left.y * point.right.x;
            }
            cv::SVD svd(A, SVD::Flags::FULL_UV);
    
            return svd.vt.row(3);
        }

        void normalizeCoordinatesByLastCol(InputArray _src, OutputArray _dst);
        
        template <typename _Tp>
        Mat lineHomographyError(Mat model, const VecMatchingPoints<_Tp> &data)
        {
            CV_Assert(traits::Type<_Tp>::value == model.type());
            Mat src = data.leftMat();
            Mat dst = data.rightMat();

            try
            {
                auto dst_H = model * src.t();
                auto src_H = model.inv() * dst.t();
                Mat resultL, resultR;
            
                normalizeCoordinatesByLastCol(src_H.t(), resultL);
                normalizeCoordinatesByLastCol(dst_H.t(), resultR);
            
                resultL = matrixVectorElementwiseMultiplication(resultL, src.col(src.cols - 1));
                resultR = matrixVectorElementwiseMultiplication(resultR, dst.col(dst.cols - 1));

                resultL = src - resultL;
                resultR = dst - resultR;

                cv::pow(resultL, 2, resultL);
                cv::pow(resultR, 2, resultR);

                Mat result = resultL + resultR;
                cv::reduce(result, result, 1, CV_REDUCE_SUM, result.type());
                cv::sqrt(result, result);
        
                return result;
            }
            catch (...)
            {
                return Mat();
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
        }


        template <typename _Tp>
        void lineRansac(int numIterations, const VecMatchingPoints<_Tp> &matchingPoints, float inlierTh = 0.35)
        {
            const int k = 3;
            auto dataSamples = VecMatchingPoints<_Tp>::randomSamples(matchingPoints, numIterations, k);
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

#endif
