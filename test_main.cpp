#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

int testOpenCVVideo()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;
    while (true)
    {
        Mat frame;
        cap >>frame;
        imshow("Webcam frame", frame);
        
        if (waitKey(30) >= 0)
            break;
    }
    return 0;
}
void ShowImage()
{
    cv::String fileName;
    fileName = R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1a.jpg)";
    auto mat = imread(fileName);
    imshow("M", mat); waitKey(0); destroyAllWindows();
}

void ShowImage(const cv::Mat &mat)
{
    imshow("M", mat); waitKey(0); destroyAllWindows();
}

string shape(const cv::Mat &mat)
{
    std::ostringstream stream;
    stream << "(" << mat.size().height << "," << mat.size().width << "," << (mat.dims + 1) << ")";
    return stream.str();
}
tuple<cv::Mat, cv::Mat> LoadImages(string img1_name, string img2_name, float scale_factor)
{
    auto img1c = imread(img1_name); // queryImage
    auto img2c = imread(img2_name); // trainImage
    
    if (scale_factor < 1)
    {
        Mat tmp;
        cout << "Image shape before scaling " << shape(img1c) << endl;
        resize(img1c, tmp, Size(), scale_factor, scale_factor, cv::INTER_AREA);
        img1c = tmp;

        resize(img2c, tmp, Size(), scale_factor, scale_factor, cv::INTER_AREA);
        img2c = tmp;
    }
    cout << "Image shape after scaling " << shape(img1c) << endl;

    cv::cvtColor(img1c, img1c, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2c, img2c, cv::COLOR_BGR2GRAY);

    return { img1c, img2c };
}



int main()
{

    //ShowImage

    cv::String img1_name = R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1a.jpg)";
    cv::String img2_name = R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1b.jpg)";
    auto t = LoadImages(img1_name, img2_name, 0.5);
    ShowImage(get<0>(t));
    ShowImage(get<1>(t));
    
    return 0;
}