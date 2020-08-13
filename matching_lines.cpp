#include "matching_lines.hpp"
#include "np_cv_imp.hpp"
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
namespace cv {
    namespace separableFundamentalMatrix {


        // Helper - Multiply matrix with vector
        vector<float> MatrixVectorMul(InputArray mat2d, Point3f vec, float scale = 1, bool absolute = false)
        {
            Mat mat = mat2d.getMat();

            vector<float> ret;
            ret.reserve(mat.size().height);

            for (size_t i = 0; i < mat.size().height; i++)
            {
                float curr = (vec.x * mat.at<float>(i, 0)) + (vec.y * mat.at<float>(i, 1)) + vec.z;
                if (absolute)
                    curr = abs(curr);
                ret.push_back(curr * scale);
            }
            return ret;
        }

        Mat createHeatmap(InputArray ptsImg1, InputArray ptsImg2, const vector<line_info> &lineInfosImg1,
            const vector<line_info> &lineInfosImg2)
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
            auto num_shared_points_vote = Indices<int>(hough_pts);

            //  Delete all non-relevent entries: That have minSharedPoints for each side (multiply by two - one for left line and one for right line).
            num_shared_points_vote.erase(
                std::remove_if(num_shared_points_vote.begin(), num_shared_points_vote.end(), [minSharedPoints](const Point3i p) {
                    return (bool)(p.x < minSharedPoints * 2);
                    }), num_shared_points_vote.end()
                        );

            // Sort the entries (in reverse order
            std::sort(num_shared_points_vote.begin(), num_shared_points_vote.end(), [](Point3i &a, Point3i &b) { return a.x > b.x; });

            // For each matching points on the matching lines,
            // project the shared points to be exactly on the line
            // start with the lines that shared the highest number of points, so we can do top-N
            // return a list index by the lines (k,j) with the projected points themself
            int num_line_ransac_iterations = int((log(0.01) / log(1 - pow(inlierRatio, 3)))) + 1;

            vector<top_line> top_lines;

            // Go over the top lines with the most number of shared points, project the points, store by the matching indices of the pair of lines
            int num_sorted_lines = min((int)num_shared_points_vote.size(), 450);
            for (size_t n = 0; n < num_sorted_lines; n++)
            {
                int k = num_shared_points_vote[n].y;
                int j = num_shared_points_vote[n].z;

                vector<Point2f> matchingPoints1;
                vector<Point2f> matchingPoints2;

                {
                    vector<int> arr_idx = intersect1d(lineInfosImg1[k].matching_indexes, lineInfosImg2[j].matching_indexes);
                    auto filteredPts = ByIndices<float>(_ptsImg1, arr_idx);
                    matchingPoints1 = projectPointsOnLine(lineInfosImg1[k].bottom_left_edge_point, lineInfosImg1[k].top_right_edge_point, filteredPts);

                    filteredPts = ByIndices<float>(_ptsImg2, arr_idx);
                    matchingPoints2 = projectPointsOnLine(lineInfosImg2[j].bottom_left_edge_point, lineInfosImg2[j].top_right_edge_point, filteredPts);
                }

                // We need at least four unique points
                vector<Point> matchingPoints1Int(matchingPoints1.begin(), matchingPoints1.end());
                vector<Point> matchingPoints2Int(matchingPoints2.begin(), matchingPoints2.end());
                //lines_info_img1

            }

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

        line_info createLineInfo(InputArray pts, const vector<float> &points_intersection, float max_distance, int line_index)
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
                matching_indexes = IndexWhereLowerThan(d, max_distance);
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

        vector<line_info> getHoughLines(InputArray pts, const int im_size_w, const int im_size_h, int min_hough_points,
            int pixel_res, int theta_res, float max_distance, int num_matching_pts_to_use)
        {
            Mat ptsRounded = pts.getMat();
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

            Mat pts1Mat = pts1.getMat();
            Mat pts2Mat = pts2.getMat();

            // we sample a small subset of features to use in the hough transform, if our sample is too sparse, increase it
            for (auto i = 0; i < top_line_retries; i++)
            {
                // rescale points and image size for fast line detection
                hough_rescale = hough_rescale * 0.5;
                max_distance_pts_line = max_distance_pts_line * 2;
                auto pts1Temp = hough_rescale * pts1Mat;
                auto pts2Temp = hough_rescale * pts2Mat;
                auto im_size_h = int(round(im_size_h_org * hough_rescale)) + 3;
                auto im_size_w = int(round(im_size_w_org * hough_rescale)) + 3;

                auto linesImg1 = getHoughLines(pts1Temp, im_size_w, im_size_h, min_hough_points, pixel_res, theta_res, max_distance_pts_line, num_matching_pts_to_use);
                auto linesImg2 = getHoughLines(pts2Temp, im_size_w, im_size_h, min_hough_points, pixel_res, theta_res, max_distance_pts_line, num_matching_pts_to_use);

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