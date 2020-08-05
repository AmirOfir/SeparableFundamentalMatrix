#include "matching_lines.hpp"
#include <opencv2\core\core_c.h>
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

// Helper - Converts an input array to vector
template<class T>
std::vector<T>& getVec(InputArray _input) {
    std::vector<T> *input;
    if (_input.isVector()) {
        input = static_cast<std::vector<T>*>(_input.getObj());
    } else {
        size_t length = _input.total();
        T* data = reinterpret_cast<T*>(_input.getMat().data);
        input = new std::vector<T>(data, data + length);
    }
    return *input;
}

// Helper - Multiply matrix with vector
vector<float> MatrixVectorMul(InputArray mat2d, Point3f vec, float scale=1, bool absolute=false)
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

vector<int> IndexWhereLowerThan(const vector<float> &vec, float maxValue)
{
    vector<int> ret;
    for (auto i = 0; i < vec.size(); i++)
    {
        if (vec[i] < maxValue)
            ret.push_back(i);
    }
    
    return ret;
}

void WhereCondition(const InputArray pts, vector<int> &firstDimension, 
    vector<int> &secondDimension, bool(*condition)(float val))
{
    Mat points = pts.getMat();

    // First dimension
    for (size_t i = 0; i < points.size().height; i++)
    {

    }

}

Mat createHeatmap(int points, const vector<line_info> &lineInfos)
{
    Mat heatmap = cv::Mat::zeros(cv::Size(points, lineInfos.size()), CV_32SC1);
    for (auto &lineInfo : lineInfos)
    {
        for (auto &matching_index : lineInfo.matching_indexes)
        {
            heatmap.at<int>(matching_index, lineInfo.line_index) = 1;
        }
    }
    return heatmap;
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
    const int heatmapSize[] = { lineInfosImg1.size(), lineInfosImg2.size(), ptsImg1.size().height };
    Mat heatmap = Mat::zeros(3, heatmapSize, CV_8U);
    
    // Fill by first lines
    for (auto &lineInfo : lineInfosImg1)
    {
        for (const int point_index : lineInfo.matching_indexes)
        {
            for (int i = 0; i < heatmapSize[1]; i++)
            {
                heatmap.at<unsigned char>(lineInfo.line_index, i, point_index) = 1;
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
                heatmap.at<unsigned char>(i, lineInfo.line_index, point_index) += 1;
            }
        }
    }

    return heatmap;
}
/* The parrallel option
#sorted_lines,lines_shared_points
def get_top_matching_lines( pts1,pts2,
                            lines_info_img1,
                            lines_info_img2,
                            min_shared_points,
                            inlier_ratio):

# The parrallel option
#     t0=timer()
#     pts_lines         = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
#     set_ix_1=set_shared_index_1(pts_lines )
#     [set_ix_1(x['matching_index'],x['line_index']) for x in lines_info_img1]
#     set_ix_2=set_shared_index_2(pts_lines)
#     [set_ix_2(x['matching_index'][0],x['line_index']) for x in lines_info_img2 if len(x['matching_index'][0])>0]
#     pts_lines[pts_lines < 2] = 0
#     hough_pts = np.sum(pts_lines, axis=0)
#     print("time is "+str(timer()-t0))


    # t0=timer()
    # pts_line1         = np.zeros((len(pts1), len(lines_info_img1)))
    # pts_line2         = np.zeros((len(pts1), len(lines_info_img2)))
    # set_ix_a=set_shared_index_a(pts_line1)
    # [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img1]
    # set_ix_a=set_shared_index_a(pts_line2)
    # [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img2]
    # print("THE time is " + str(timer() - t0))
    # pts_lines = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
    # a, b = np.where(pts_line1 > 0)
    # pts_lines[a, b, :] = 1
    # a, b = np.where(pts_line2 > 0)
    # pts_lines[a, :, b] += 1
    # print("THE time is " + str(timer() - t0))
    # # First remove all entries which does not have two matching lines
    # pts_lines[pts_lines < 2] = 0
    # # Sum across points' index, this gives us how many shared points for each pair of lines
    # hough_pts = np.sum(pts_lines, axis=0)
    # # print("the time is 1" + str(timer() - t0))
    */

template <typename TSource, typename TDest>
void reduce3d(InputArray _src, OutputArray _dst, int dim, int rtype, int dtype)
{
    CV_Assert( _src.dims() == 3 );
    CV_Assert( dim <= 2 && dim >= 0);
    Mat src = _src.getMat();

    // Create dst
    int sizes[2];
    if (dim == 0)
    {
        sizes[0] = src.size[1];
        sizes[1] = src.size[2];
    }
    else if (dim == 1)
    {
        sizes[0] = src.size[0];
        sizes[1] = src.size[2];
    }
    else
    {
        sizes[0] = src.size[0];
        sizes[1] = src.size[1];
    };
    _dst.create(2, sizes, dtype);
    Mat dst = _dst.getMat();
    
    // Fill
    int reduce_count = src.size[dim];
    parallel_for_(Range(0, sizes[0]), [&](const Range& range) {

            for (int i = range.start; i < range.end; i++)
            {
                for (int j = 0; j < sizes[1]; j++)
                {
                    TDest c = 0;
                    for (size_t k = 0; k < reduce_count; k++)
                    {
                        TSource s = src.at<TSource>(dim == 0 ? k : i, dim == 1 ? k : j, dim == 2 ? k : j);
                        if (rtype == CV_REDUCE_SUM || rtype == CV_REDUCE_AVG)
                            c += s;
                        else if (rtype == CV_REDUCE_MAX)
                            c = max(s, c);
                        else if (rtype == CV_REDUCE_MIN)
                            c = min(s, c);
                    }

                    dst.at<TDest>(i, j) = c;
                }
            }
        });
}

void reduce3d(InputArray _src, OutputArray _dst)
{
    CV_Assert( _src.dims() == 3 );
    Mat src = _src.getMat();

    // Create dst
    int sizes[]{ src.size[0], src.size[1] };
    _dst.create(2, sizes, CV_32S);
    Mat dst = _dst.getMat();
    
    // Fill
    int reduce_count = src.size[2];
    parallel_for_(Range(0, sizes[0]), [&](const Range& range) {
        
        for (int i = range.start; i < range.end; i++)
        {
            for (int j = 0; j < sizes[1]; j++)
            {
                int c = 0;
                for (size_t k = 0; k < reduce_count; k++)
                {
                    c += src.at<unsigned char>( i, j, k );
                }
                dst.at<int>(i, j) = c;
            }
        }
        });
}

void getTopMatchingLines(InputArray ptsImg1, InputArray ptsImg2, const vector<line_info> &lineInfosImg1,
    const vector<line_info> &lineInfosImg2, int minSharedPoints, float inlierRatio)
{
    // Create a heatmap between points of each line
    Mat heatmap = createHeatmap(ptsImg1, ptsImg2, lineInfosImg1, lineInfosImg2); 

    // Remove all entries which does not have two matching lines
    //pts_lines[pts_lines<2] =0
    heatmap.setTo(0, heatmap < 2);

    Mat heatmap_summed;
    reduce3d(heatmap, heatmap_summed);
    //reduce3d<unsigned char, int>(heatmap, heatmap_summed, 2, CV_REDUCE_SUM, CV_32S);
    int x;
}
// The fastest option
/*
#sorted_lines,lines_shared_points
def get_top_matching_lines( pts1,pts2,
                            lines_info_img1,
                            lines_info_img2,
                            min_shared_points,
                            inlier_ratio):

    # The fastest option
#     t0=timer()
    pts_line1a         = np.zeros((len(pts1), len(lines_info_img1)))
    pts_line2b         = np.zeros((len(pts1), len(lines_info_img2)))
    set_ix_a=set_shared_index_single(pts_line1a)
    [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img1]
    set_ix_a=set_shared_index_single(pts_line2b)
    [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img2]
    a1, b1 = np.where(pts_line1a > 0)
    a2, b2 = np.where(pts_line2b > 0)
    pts_lines = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
    set_index_both=set_shared_index_both(pts_lines)
    [set_index_both(x[0], b1[np.where(a1==x[0])], x[1]) for x in zip(a2,b2) if x[0] in a1]
    [set_index_both(x[0], x[1], b2[np.where(a2==x[0])]) for x in zip(a1,b1) if x[0] in a2]
    hough_pts = sum(pts_lines)
    # print("the time is 0 " + str(timer() - t0))


    # convert to a list where each entry is 1x3: the number of shared points for each pair of line and their indices
    n_lines1,n_lines2=np.shape(hough_pts)
    r, c      = np.indices((n_lines1, n_lines2))
    dim_lines = n_lines1*n_lines2

    # use voting to find out which lines shares points
    num_shared_points_vote=np.concatenate((np.concatenate((hough_pts[r.reshape(dim_lines,1),c.reshape(dim_lines,1)],
                                                           r.reshape(dim_lines,1)),axis=1),c.reshape(dim_lines,1)),axis=1)



    # sort the entries, but delete all non-relevent before to save space and time
    # all entries are at least two (one for each left and one for right line) so we multiply it by two
    num_shared_points_vote= np.delete(num_shared_points_vote, (np.where(num_shared_points_vote<min_shared_points*2)), axis=0)
    sorted_lines          = sorted(np.array(num_shared_points_vote), key=lambda X: X[0], reverse=True)



    # For each matching points on the matching lines,
    # project the shared points to be exactly on the line
    # start with the lines that shared the highest number of points, so we can do top-N
    # return a list index by the lines (k,j) with the projected points themself
    num_line_ransac_iterations =int((np.log(0.01) / np.log(1 - inlier_ratio ** 3)))+1
    top_lines = []
    # Go over the top lines with the most number of shared points, project the points, store by the matching indices of the pair of lines
    num_sorted_lines=np.min((len(sorted_lines),450))
    for n in range(num_sorted_lines):
        k=int(sorted_lines[n][1])
        j=int(sorted_lines[n][2])

        # some points might be too near each other
        arr_idx = np.intersect1d(lines_info_img1[k]['matching_index'][0], lines_info_img2[j]['matching_index'][0])

        (x1a,y1a,x2a,y2a)=lines_info_img1[k]['edge_points']
        (x1b,y1b,x2b,y2b)=lines_info_img2[j]['edge_points']
        matching_points1=[]
        matching_points2=[]

        arr_shared_pts=np.concatenate((pts1[arr_idx,:],pts2[arr_idx,:]),axis=1)
        # project the points
        for pt in arr_shared_pts:

            nearest_pt_on_line1=project_point_line(np.array((x1a, y1a)).astype(float), np.array((x2a, y2a)).astype(float), np.array((pt[0], pt[1])).astype(float))
            nearest_pt_on_line2=project_point_line(np.array((x1b, y1b)).astype(float), np.array((x2b, y2b)).astype(float), np.array((pt[2], pt[3])).astype(float))

            matching_points1.append(nearest_pt_on_line1)
            matching_points2.append(nearest_pt_on_line2)


        matching_points1=np.array(matching_points1)
        matching_points2=np.array(matching_points2)

        # we need at least four unique points
        _,unique_idx_1=np.unique(matching_points1.astype(int),axis=0,return_index=True)
        _,unique_idx_2=np.unique(matching_points2.astype(int),axis=0,return_index=True)
        unique_idx=np.intersect1d(unique_idx_1,unique_idx_2)

        if len(unique_idx)<4:
            continue

        matching_points1=matching_points1[unique_idx, :]
        matching_points2=matching_points2[unique_idx, :]

        # Find inliers, inlier_idx_homography - index of inliers of all the line points
        inlier_idx_homography,hom_average_err = ransac_get_line_inliers(int(num_line_ransac_iterations),matching_points1,matching_points2)
        if len(inlier_idx_homography)<4:
            continue

        inlier_points = np.array(matching_points1)[inlier_idx_homography]
        inlier_selected_idx, inlier_selected_max_dist, inlier_selected_min_dist = select_max_dist(inlier_points)


        mp1 = np.array(matching_points1)[inlier_idx_homography][inlier_selected_idx]
        mp2 = np.array(matching_points2)[inlier_idx_homography][inlier_selected_idx]

        # should also take the ones with the most distant points
        top_lines.append(
            {'num_inliers'  :len(inlier_idx_homography),
             'line_points_1':np.array(matching_points1)[inlier_idx_homography],
             'line_points_2':np.array(matching_points2)[inlier_idx_homography],
             'line1_index':k,
             'line2_index':j,
             'inlier_selected_index':inlier_selected_idx,
             'selected_line_points1':mp1,
             'selected_line_points2':mp2,
             'max_dist':inlier_selected_max_dist,
             'min_dist':inlier_selected_min_dist,
             'homg_err':hom_average_err})




    if len(top_lines)<2:
        return [],[],[]
    top_lines.sort(key=lambda X: X['min_dist'],reverse=True) # max_dist
    top_two_lines=[None,None]
    top_two_lines[0]=top_lines[0]
    a=lines_info_img1[top_lines[0]['line1_index']]['line_eq_abc_norm'][0]
    b=lines_info_img1[top_lines[0]['line1_index']]['line_eq_abc_norm'][1]
    line_eqs= [lines_info_img1[x['line1_index']]['line_eq_abc_norm'] for x in top_lines[1:20]]
    all_angles=[ np.arccos(np.min((a*x[0]+b*x[1],1)))*180 / np.pi for x in line_eqs]
    max_angle_ix=np.argmax(np.min(np.array((np.array(all_angles),180-np.array(all_angles))),axis=0))
    top_two_lines[1] = top_lines[max_angle_ix]
    return top_two_lines,lines_info_img1,lines_info_img2
*/


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
    
    Point3f pt1(points_intersection[0],points_intersection[1], 1);
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
        Mat A(2,2, CV_8U, &a);
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

void FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2, 
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
