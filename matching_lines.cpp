#include "matching_lines.hpp"
#include "np_cv_imp.hpp"
#include "line_geometry.hpp"
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
namespace cv {
    namespace separableFundamentalMatrix {
        


        // Helper - Multiply matrix with vector
        vector<float> MatrixVectorMul(Mat mat2d, Point3f vec, float scale = 1, bool absolute = false)
        {
            vector<float> ret;
            ret.reserve(mat2d.size().height);

            for (size_t i = 0; i < mat2d.size().height; i++)
            {
                float curr = (vec.x * mat2d.at<float>(i, 0)) + (vec.y * mat2d.at<float>(i, 1)) + vec.z;
                if (absolute)
                    curr = abs(curr);
                ret.push_back(curr * scale);
            }
            return ret;
        }

        Mat createHeatmap(InputArray ptsImg1, InputArray ptsImg2, const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2)
        {
            /*
            pts_lines         = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))

            set_ix_1=set_shared_index_1(pts_lines)
            [set_ix_1(x['matching_index'],x['line_index']) for x in lines_info_img1]
            set_ix_2=set_shared_index_2(pts_lines)
            [set_ix_2(x['matching_index'][0],x['line_index']) for x in lines_info_img2 if len(x['matching_index'][0])>0]

            */

            // Create heatmap
            const int heatmapSize[] = { (int)lineInfosImg1.size(), (int)lineInfosImg2.size(), ptsImg1.size().height };
            Mat heatmap = Mat::zeros(3, heatmapSize, CV_8U);

            // Fill by first lines
            for (auto &lineInfo : lineInfosImg1)
            {
                for (const int point_index : lineInfo.matching_indexes)
                {
                    for (int i = 0; i < heatmapSize[1]; i++)
                    {
                        heatmap.at<uchar>(lineInfo.line_index, i, point_index) = 1;
                    }
                }
            }

            // Fill by second lines
            for (auto &lineInfo : lineInfosImg2)
            {
                for (const int point_index : lineInfo.matching_indexes)
                {
                    for (int i = 0; i < heatmapSize[0]; i++)
                    {
                        heatmap.at<uchar>(i, lineInfo.line_index, point_index) += 1;
                    }
                }
            }

            return heatmap;
        }

        vector<Point2f> projectPointsOnLineByInices(InputArray _pts, const line_info &lineInfo, vector<int> indices)
        {
            vector<Point2f> filteredPts;
            filteredPts = byIndices<float>(_pts, indices);
            return projectPointsOnLine(lineInfo.bottom_left_edge_point, lineInfo.top_right_edge_point, filteredPts);
        }

        vector<int> uniqueIntersectedPoints(const vector<Point2f> &matchingPoints1, const vector<Point2f> &matchingPoints2)
        {
            vector<size_t> uniqueIdx1 = index_unique(vector<Point>(matchingPoints1.begin(), matchingPoints1.end()));
            vector<size_t> uniqueIdx2 = index_unique(vector<Point>(matchingPoints2.begin(), matchingPoints2.end()));

            vector<int> unique_idx;
            intersect1d(uniqueIdx1.begin(), uniqueIdx1.end(), uniqueIdx2.begin(), uniqueIdx2.end(), back_inserter(unique_idx));

            return unique_idx;
        }

        void getTopMatchingLines(InputArray _ptsImg1, InputArray _ptsImg2, const vector<line_info> &lineInfosImg1,
            const vector<line_info> &lineInfosImg2, int minSharedPoints, float inlierRatio)
        {

            // Create a heatmap between points of each line
            Mat heatmap = createHeatmap(_ptsImg1, _ptsImg2, lineInfosImg1, lineInfosImg2);

            // Remove all entries which does not have two matching lines (pts_lines[pts_lines<2] =0)
            heatmap.setTo(0, heatmap < 2);

            // Sum across points' index, this gives us how many shared points for each pair of lines
            Mat hough_pts;
            reduceSum3d<uchar, int>(heatmap, hough_pts, (int)CV_32S);

            // Use voting to find out which lines shares points
            // Convert to a list where each entry is 1x3: the number of shared points for each pair of line and their indices
            auto num_shared_points_vote = indices<int>(hough_pts);

            //  Delete all non-relevent entries: That have minSharedPoints for each side (multiply by two - one for left line and one for right line).
            num_shared_points_vote.erase(
                std::remove_if(num_shared_points_vote.begin(), num_shared_points_vote.end(), [minSharedPoints](const Point3i p) {
                    return (bool)(p.x < minSharedPoints * 2);
                    }), num_shared_points_vote.end()
                        );

            // Sort the entries (in reverse order)
            // Note: could've sorted them by x only, but the python code sorted like that
            //std::sort(num_shared_points_vote.begin(), num_shared_points_vote.end(), lexicographicalSort3d<int>);
            std::sort(num_shared_points_vote.begin(), num_shared_points_vote.end(), 
                [](Point3i &a, Point3i &b)
                {
                    return a.x > b.x || (a.x == b.x && a.y < b.y) || (a.x ==b.x && a.y==b.y && a.z < b.z);
                });

            // For each matching points on the matching lines,
            // project the shared points to be exactly on the line
            // start with the lines that shared the highest number of points, so we can do top-N
            // return a list index by the lines (k,j) with the projected points themself
            int num_line_ransac_iterations = int((log(0.01) / log(1 - pow(inlierRatio, 3)))) + 1;
            
            vector<top_line> topLines;

            // Go over the top lines with the most number of shared points, project the points, store by the matching indices of the pair of lines
            int num_sorted_lines = min((int)num_shared_points_vote.size(), 450);
            for (size_t n = 0; n < num_sorted_lines; n++)
            {
                int k = num_shared_points_vote[n].y;
                int j = num_shared_points_vote[n].z;

                vector<int> arr_idx;
                intersect1d(lineInfosImg1[k].matching_indexes.begin(), lineInfosImg1[k].matching_indexes.end(), 
                    lineInfosImg2[j].matching_indexes.begin(), lineInfosImg2[j].matching_indexes.end(), back_inserter(arr_idx));

                vector<Point2f> matchingPoints1 = projectPointsOnLineByInices(_ptsImg1, lineInfosImg1[k], arr_idx);
                vector<Point2f> matchingPoints2 = projectPointsOnLineByInices(_ptsImg2, lineInfosImg2[j], arr_idx);

                vector<int> uniqueIdx = uniqueIntersectedPoints(matchingPoints1, matchingPoints2);

                // We need at least four unique points
                if (uniqueIdx.size() < 4)
                    continue;

                // Filter
                matchingPoints1 = byIndices<float>(matchingPoints1, uniqueIdx);
                matchingPoints2 = byIndices<float>(matchingPoints2, uniqueIdx);

                // Find inliers, inlier_idx_homography - index of inliers of all the line points
                auto matchingPoints = VecMatchingPoints<float>(matchingPoints1, matchingPoints2);
                auto lineInliersResult = lineInliersRansac(num_line_ransac_iterations, matchingPoints);

                if (lineInliersResult.inlierIndexes.size() < 4)
                    continue;

                auto inlierPoints1 = byIndices<float>(matchingPoints1, lineInliersResult.inlierIndexes);
                auto inlierPoints2 = byIndices<float>(matchingPoints1, lineInliersResult.inlierIndexes);
                
                auto endpoints = intervalEndpoints(inlierPoints1);
                auto median = intervalMedian(inlierPoints1, endpoints.firstIdx, endpoints.secondIdx);
                
                top_line curr;
                curr.num_inliers = (int)lineInliersResult.inlierIndexes.size();
                curr.line_points_1 = inlierPoints1;
                curr.line_points_2 = inlierPoints2;
                curr.line1_index = k;
                curr.line2_index = j;
                curr.inlier_selected_index = { endpoints.firstIdx, endpoints.secondIdx, median.medianIdx };
                curr.selected_line_points1 = byIndices<float>(inlierPoints1, curr.inlier_selected_index);
                curr.selected_line_points2 = byIndices<float>(inlierPoints2, curr.inlier_selected_index);
                curr.max_dist = endpoints.distance;
                curr.min_dist = median.minDistance;
                curr.homg_err = lineInliersResult.meanError;
                topLines.push_back(curr);

            }
            
            if (topLines.size() < 2)
                return;
            sort(topLines.begin(), topLines.end(),
                [](const top_line &line1, const top_line &line2) { return line1.min_dist <= line2.min_dist; });

            top_line topTwoLines[2];
            topTwoLines[0] = topLines[0];
            //auto a = lineInfosImg1[topTwoLines[0].line1_index].line_eq_abc_norm.x;

            //return topLines;
        }

        vector<float> findIntersectionPoints(float rho, float theta, const int im_size_w, const int im_size_h)
        {
            float a = cos(theta);
            float b = sin(theta);
            float x_0 = a != 0 ? rho / a : -1000;
            float x_1 = a != 0 ? (rho - (b * im_size_w)) / a : -1000;
            float y_0 = b != 0 ? rho / b : -1000;
            float y_1 = b != 0 ? (rho - (a * im_size_h)) / b : -1000;

            vector<float> ret;
            if (x_0 >= 0 && x_0 < im_size_h)
            {
                ret.push_back(x_0);
                ret.push_back(0);
            }
            if (y_0 >= 0 && y_0 < im_size_w)
            {
                ret.push_back(0);
                ret.push_back(y_0);
            }
            if (x_1 >= 0 && x_1 < im_size_h)
            {
                ret.push_back(x_1);
                ret.push_back(float(im_size_w));
            }
            if (y_1 >= 0 && y_1 < im_size_w)
            {
                ret.push_back(float(im_size_h));
                ret.push_back(y_1);
            }

            return ret;
        }

        line_info createLineInfo(Mat pts, const vector<float> &points_intersection, float max_distance, int line_index)
        {
            CV_Assert(points_intersection.size() == 4);

            Point3f pt1(points_intersection[0], points_intersection[1], 1);
            Point3f pt2(points_intersection[2], points_intersection[3], 1);
            Point3f line_eq = pt1.cross(pt2);

            if (abs(line_eq.z) >= FLT_EPSILON)
            {
                // divide by z
                line_eq /= line_eq.z;
            }
            // too small to divide, solve with least square
            else
            {
                float a[4] = { points_intersection[0],1,
                               points_intersection[2],1 };
                Mat A(2, 2, CV_8U, &a);
                vector<float> B{ points_intersection[1], points_intersection[2] };
                vector<float> x;
                solve(A, B, x);
                line_eq.x = x[0];
                line_eq.y = -1;
                line_eq.z = 0;
            }

            vector<int> matching_indexes;
            {
                float scale = sqrtf((line_eq.x * line_eq.x) + (line_eq.y * line_eq.y));
                vector<float> d = MatrixVectorMul(pts, line_eq, 1.f / scale, true);
                matching_indexes = index_if(d.begin(), d.end(), [&](float f) {return f < max_distance; });
            }

            auto lineEqNormDivider = sqrt(pow(line_eq.x, 2) + pow(line_eq.y, 2)) + FLT_EPSILON;

            line_info ret;
            ret.matching_indexes = matching_indexes;
            ret.line_eq_abc = line_eq;
            ret.line_eq_abc_norm = line_eq / lineEqNormDivider;
            ret.bottom_left_edge_point = Point2f(points_intersection[0], points_intersection[1]);
            ret.top_right_edge_point = Point2f(points_intersection[2], points_intersection[3]);
            ret.max_distance = max_distance;
            ret.line_index = line_index;
            return ret;
        }

        vector<line_info> getHoughLines(Mat pts, const int im_size_w, const int im_size_h, int min_hough_points,
            int pixel_res, int theta_res, float max_distance, int num_matching_pts_to_use)
        {
            Mat ptsRounded = pts.clone();
            ptsRounded.convertTo(ptsRounded, CV_32S);

            Mat bw_img = Mat::zeros(im_size_h, im_size_w, CV_8U);
            num_matching_pts_to_use = min(ptsRounded.size().height, num_matching_pts_to_use);
            for (int addedCount = 0; addedCount < num_matching_pts_to_use; ++addedCount)
            {
                int x0 = ptsRounded.at<int>(addedCount, 1), x1 = ptsRounded.at<int>(addedCount, 0);
                bw_img.at<uint8_t>(x0, x1) = (unsigned short)255;
            }

            vector<Vec2f> houghLines;
            cv::HoughLines(bw_img, houghLines, pixel_res, CV_PI / theta_res, min_hough_points);

            vector<line_info> lineInfos;
            int lineIndex = 0;
            for (auto l : houghLines)
            {
                float rho = l[0], theta = l[1];
                auto p_intersect = findIntersectionPoints(rho, theta, im_size_w, im_size_h);
                if (p_intersect.size() == 4)
                {
                    lineInfos.push_back(createLineInfo(pts, p_intersect, max_distance, lineIndex));
                    ++lineIndex;
                }
            }

            return lineInfos;
        }

        void cv::separableFundamentalMatrix::FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
            const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
            int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio)
        {
            hough_rescale = hough_rescale * 2; // for the first time
            max_distance_pts_line = max_distance_pts_line * 0.5;

            Mat pts1Org = pts1.isMat() ? pts1.getMat() : pts1.getMat().t();
            Mat pts2Org = pts2.isMat() ? pts2.getMat() : pts2.getMat().t();

            // we sample a small subset of features to use in the hough transform, if our sample is too sparse, increase it
            for (auto i = 0; i < top_line_retries; i++)
            {
                // rescale points and image size for fast line detection
                hough_rescale = hough_rescale * 0.5;
                max_distance_pts_line = max_distance_pts_line * 2;
                auto pts1 = hough_rescale * pts1Org;
                auto pts2 = hough_rescale * pts2Org;
                auto im_size_h = int(round(im_size_h_org * hough_rescale)) + 3;
                auto im_size_w = int(round(im_size_w_org * hough_rescale)) + 3;

                auto linesImg1 = getHoughLines(pts1, im_size_w, im_size_h, min_hough_points, pixel_res, theta_res, max_distance_pts_line, num_matching_pts_to_use);
                auto linesImg2 = getHoughLines(pts2, im_size_w, im_size_h, min_hough_points, pixel_res, theta_res, max_distance_pts_line, num_matching_pts_to_use);

                if (linesImg1.size() && linesImg2.size())
                {
                    getTopMatchingLines(pts1, pts2, linesImg1, linesImg2, min_shared_points, inlier_ratio);
                }
            }
            /*
                if len(lines_img1)!=0 and len(lines_img2) !=0:
                    top_lines_by_inliers,lines1,lines2 = get_top_matching_lines(pts1, pts2, lines_img1, lines_img2,min_shared_points,inlier_ratio)

                if len(top_lines_by_inliers)>=2: # We found the lines
                    break

                print("Sep F 4pts:Rescale again")
                */
        }

    }
}