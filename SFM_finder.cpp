#include "SFM_finder.hpp"
#include "matching_lines.hpp"
#include "sfm_ransac.hpp"

using namespace cv::separableFundamentalMatrix;

// pts1 is Mat of shape(X,2)
// pts2 is Mat of shape(X,2)
Mat cv::separableFundamentalMatrix::findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
        float inlier_ratio, int inlier_threshold,
        float hough_rescale, int num_matching_pts_to_use, int pixel_res, int min_hough_points,
        int theta_res, float max_distance_pts_line, int top_line_retries, int min_shared_points)
{
    int pts1Count = pts1.isVector() ? pts1.getMat().size().width : pts1.getMat().size().height;
    if (hough_rescale == DEFAULT_HOUGH_RESCALE)
        hough_rescale = float(2 * pts1Count) / im_size_h_org;
    else if (hough_rescale > 1) // Only subsample
        hough_rescale = 1;

    auto topMatchingLines = FindMatchingLines(im_size_h_org, im_size_w_org, pts1, pts2, top_line_retries, hough_rescale, max_distance_pts_line, min_hough_points, pixel_res, theta_res, num_matching_pts_to_use, min_shared_points, inlier_ratio);

    // We have at least one line
    if (topMatchingLines.size())
    {
        Mat data = prepareDataForRansac(pts1, pts2);
        vector<Mat> lines = prepareLinesForRansac(topMatchingLines);
        uint maxNumIterations = (uint)std::floor(1 + std::log(0.01) / std::log(1 - pow(inlier_ratio, 5)));
    }

    return Mat();
}