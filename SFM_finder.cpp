#include "SFM_finder.hpp"
#define DEFAULT_HOUGH_RESCALE -1

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
vector<float> MatrixVectorMul(InputArray mat2d, Point3f vec, float divideByConstant=1)
{
    Mat mat = mat2d.getMat();
    
    vector<float> ret;
    ret.reserve(mat.size().height);

    for (size_t i = 0; i < mat.size().height; i++)
    {
        float curr = (vec.x * mat.at<float>(i, 0)) + (vec.y * mat.at<float>(i, 1)) + vec.z;
        ret.push_back(curr / divideByConstant);
    }
    return ret;


}
OutputArray findIntersectionPoints(float rho, float theta, const int im_size_w, const int im_size_h)
{
    float a = cos(theta);
    float b = sin(theta);
    float x_0 = a != 0 ? rho / a : -1000;
    float x_1 = a != 0 ? rho - (b * im_size_h) : -1000;
    float y_0 = b != 0 ? rho / b : -1000;
    float y_1 = b != 0 ? rho - (a*im_size_w) : -1000;
    
    vector<float> ret;
    if (x_0 >= 0 && x_0 < im_size_w) 
    {
        ret.push_back(x_0);
        ret.push_back(0);
    }
    if (y_0 >= 0 && y_0 < im_size_h)
    {
        ret.push_back(0);
        ret.push_back(y_0);
    }
    if (x_1 >= 0 && x_1 < im_size_w)
    {
        ret.push_back(x_1);
        ret.push_back(float(im_size_h));
    }
    if (y_1 >= 0 && y_1 < im_size_h)
    {
        ret.push_back(float(im_size_w));
        ret.push_back(y_1);
    }

    return ret;
}

struct line_info
{
    vector<int> matching_index;
    float line_eq_abc;
    float line_eq_abc_norm;
    Vec2f bottom_left_edge_point;
    Vec2f top_right_edge_point;
    float max_distance;
    int line_index;
};


line_info createLineInfo(InputArray pts, const InputArray points_intersection, float max_distance, int line_index)
{
    CV_Assert(points_intersection.size().width == 4);
    
    vector<float> points = getVec<float>(points_intersection);
    Point3f pt1(points[0],points[1], 1);
    Point3f pt2(points[2], points[3], 1);
    Point3f line_eq = pt1.cross(pt2);

    
    if (abs(line_eq.z) >= FLT_EPSILON)
    {
        // divide by z
        line_eq /= line_eq.z;
    }
    // too small to divide, solve with least square
    else
    {
        float a[4] = { points[0],1,
                       points[2],1 };
        Mat A(2,2, CV_8U, &a);
        vector<float> B{ points[1], points[2] };
        vector<float> x;
        solve(A, B, x);
        line_eq.x = x[0];
        line_eq.y = -1;
        line_eq.z = 0;
    }

    float divider = sqrtf((line_eq.x * line_eq.x) + (line_eq.y * line_eq.y));
    vector<float> d = MatrixVectorMul(pts, line_eq, divider);
    
}
/*def line_info(pts,points_intersection,max_distance,line_index):
    x1, y1, x2, y2 = points_intersection
    line_eq = np.cross([x1, y1, 1], [x2, y2, 1])
    if (abs(line_eq[2])) < np.finfo(float).eps :  # too small to divide, solve with least square
        points = [(x1, y1), (x2, y2)]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        # y= ax +c
        a, c = np.linalg.lstsq(A, y_coords)[0]
        # c should be zero
        c = 0
        b = -1
        line_eq=np.array([a,b,c])
    else:
        line_eq = line_eq / line_eq[2]
        (a, b, c) = line_eq

    d = abs((a * pts[:, 0] + b * pts[:, 1] + c)) / (math.sqrt(a * a + b * b))
    return {'matching_index': np.where(d < max_distance),
            'line_eq_abc' : line_eq,
            'line_eq_abc_norm': line_eq/np.sqrt((line_eq[0] ** 2 + line_eq[1] ** 2)+1e-17), # normalize coor. for angles between lines
            'edge_points' : (x1, y1, x2, y2),
            'max_distance': max_distance,
            'line_index'  : line_index}
            */
void getHoughLines(InputArray pts, const int im_size_w, const int im_size_h, int min_hough_points,
    int pixel_res, int theta_res, float max_distance, int num_matching_pts_to_use)
{
    Mat ptsRounded = pts.getMat();
    ptsRounded.convertTo(ptsRounded, CV_32S);

    Mat bw_img = Mat(im_size_h, im_size_w, CV_8U);
    num_matching_pts_to_use = pts.cols() < num_matching_pts_to_use ? pts.cols() : num_matching_pts_to_use;
    for (int addedCount = 0; addedCount < num_matching_pts_to_use; ++addedCount)
    {
        bw_img.at<uint8_t>(ptsRounded.at<int>(addedCount, 0), ptsRounded.at<int>(addedCount, 1)) = (unsigned short)255;
    }
    
    vector<Vec2f> lines;
    cv::HoughLines(bw_img, lines, pixel_res, CV_PI / theta_res, min_hough_points);

    int line_index = 0;
    for (auto l : lines)
    {
        float rho = l[0], theta = l[1];
        auto p_intersect = findIntersectionPoints(rho, theta, im_size_w, im_size_h);
        if (p_intersect.size().width == 4)
        {

        }
    }

    /*
    
    line_index=0
    D_lines_1=[]
    for l in lines1:
        rho, theta = l[0]
        p_intersect = find_intersection_points(rho,theta,img_size)
        if len(p_intersect)==4:
            D_lines_1.append(line_info(pts1,p_intersect,max_distance,line_index))
            line_index=line_index+1

    D_lines_2 = []
    line_index=0
    for l in lines2:
        rho, theta = l[0]
        p_intersect = find_intersection_points(rho,theta,img_size)
        if len(p_intersect)==4:
            D_lines_2.append(line_info(pts2,p_intersect,max_distance,line_index))
            line_index=line_index+1

    return D_lines_1,D_lines_2
*/
}

void FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2, 
    const int top_line_retries, int hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res, 
    int theta_res, int num_matching_pts_to_use)
{
    if (hough_rescale == DEFAULT_HOUGH_RESCALE)
        hough_rescale = (2 * pts1.size().height) / im_size_h_org;
    else if (hough_rescale > 1) // Only subsample
        hough_rescale = 1;

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
        auto pts1Temp = (float)hough_rescale * pts1Mat;
        auto pts2Temp = (float)hough_rescale * pts2Mat;
        auto im_size_h = int(round(im_size_h_org * hough_rescale)) + 3;
        auto im_size_w = int(round(im_size_w_org * hough_rescale)) + 3;
        
        
        lines_img1, lines_img2 = getHoughLines(im_size_h,im_size_w, pts1, pts2, min_hough_points, pixel_res, 
            theta_res, max_distance_pts_line, num_matching_pts_to_use)
    }
    /*
    top_lines_by_inliers=[]
    lines1=[]
    lines2=[]
    hough_rescale         = hough_rescale * 2 # for the first time
    max_distance_pts_line = max_distance_pts_line * 0.5
    # we sample a small subset of features to use in the hough transform, if our sample is too sparse, increase it
    for _ in range(top_line_retries):
        # rescale points and image size for fast line detection
        hough_rescale         = hough_rescale * 0.5
        max_distance_pts_line = max_distance_pts_line * 2
        pts1 = pts1org * hough_rescale
        pts2 = pts2org * hough_rescale
        im_size_h = int(round(im_size_h_org * hough_rescale)) + 3
        im_size_w = int(round(im_size_w_org * hough_rescale)) + 3

        lines_img1, lines_img2 = get_hough_lines(np.array([im_size_h,im_size_w]),
                                                 pts1,
                                                 pts2,
                                                 min_hough_points, pixel_res, theta_res,max_distance_pts_line,num_matching_pts_to_use)

        if len(lines_img1)!=0 and len(lines_img2) !=0:
            top_lines_by_inliers,lines1,lines2 = get_top_matching_lines(pts1, pts2, lines_img1, lines_img2,min_shared_points,inlier_ratio)

        if len(top_lines_by_inliers)>=2: # We found the lines
            break

        print("Sep F 4pts:Rescale again")
        */
}

// pts1 is vector<int>
// pts2 is vector<int>
Mat findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
    float inlier_ratio = 0.4, int inlier_threshold = 3,
    int hough_rescale = -1, int num_matching_pts_to_use = 150, int pixel_res = 4, int min_hough_points = 4,
    int theta_res = 180, float max_distance_pts_line = 3, int top_line_retries = 2, int min_shared_points = 4)
{
    
    return Mat();
}