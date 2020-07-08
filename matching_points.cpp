#include "matching_points.hpp"
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
void MatchPoints(const Mat& img1c, const Mat& img2c, vector<int> &pts1, vector<int> &pts2)
{
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create()


    BFMatcher bf;
    vector<vector<DMatch>> matches;
    //bf.knnMatch()

    

    /*for (size_t i = 0; i < matches.size(); i++)
    {
        for (size_t i = 0; i < length; i++)
        {

        }
        if (matches[i][0].distance)
    }
*/



    /*
    descriptor = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = descriptor.detectAndCompute(img1,None)
    kp2, desc2 = descriptor.detectAndCompute(img2,None)

    pts1all = np.float32([kp.pt for kp in kp1])
    pts2all = np.float32([kp.pt for kp in kp2])

    
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(desc1,desc2,k=2)
    all_matches = sorted(all_matches, key=lambda x: x[0].distance)
    all_matches =all_matches[0:1000]
    matches = []

    for m, n in all_matches:
        if m.distance < 0.8 * n.distance:
            matches.append([m])
    pts1 = np.float32([pts1all[m[0].queryIdx] for m in matches])
    pts2 = np.float32([pts2all[m[0].trainIdx] for m in matches])

    */


}