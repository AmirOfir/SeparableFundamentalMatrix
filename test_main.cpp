#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include "SFM_finder.hpp"
#include "test_precomp.hpp"

using namespace cv;
using namespace cv::separableFundamentalMatrix;
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

vector<vector<float>> parseCSV(string csv_name)
{
    ifstream data(csv_name);
    string line;
    std::string::size_type sz;
    vector<vector<float>> parsedCsv;
    while(std::getline(data,line))
    {
        stringstream lineStream(line);
        string cell;
        vector<float> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(stof(cell, &sz));
        }

        parsedCsv.push_back(parsedRow);
    }

    return parsedCsv;
};

template <typename RealType>
Mat convertVectorOfVectorsToMat(const vector<vector<RealType>> &vec, int chs = 1)
{
    Mat mat(vec.size(), vec.at(0).size(), CV_MAKETYPE(cv::DataType<RealType>::type, chs));
    for(int i=0; i<mat.rows; ++i)
         for(int j=0; j<mat.cols; ++j)
              mat.at<RealType>(i, j) = vec.at(i).at(j);
    return mat;
}

template <typename RealType>
vector<Point2f> convertToVectorOfPoints(const vector<vector<RealType>> &vec, int chs = 1)
{
    vector<Point2f> ret;
    for (auto c : vec)
        ret.push_back(Point2f(c.at(0), c.at(1)));
    return ret;
}
 
void LoadImages(string img1_name, string img2_name, string imgA_pts_name, string imgB_pts_name, float scale_factor, 
    Mat &img1, Mat &img2, vector<vector<float>> &ptsA, vector<vector<float>> &ptsB)
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

    ptsA = parseCSV(imgA_pts_name);
    ptsB = parseCSV(imgB_pts_name);
    img1 = img1c;
    img2 = img2c;
}

int main()
{
    //test_intervalEndpoints();
    //test_intervalMedian();

    //test_maxDistance();
    //test_findLineHomography();
    //test_lineInliersRansac();
    //test_lineHomographyError();
    //test_normalizeCoordinatesByLastCol();
    //test_matrixVectorElementwiseMultiplication();
    //return 0;
    vector<tuple<string, string, string, string, float>> example_files = 
    {
        {
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1a.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1b.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts1a.csv)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts1b.csv)",
            0.5
        },
        {
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im2a.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im2b.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts2a.csv)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts2b.csv)", 1
        },
        {
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im3a.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im3b.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts3a.csv)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts3b.csv)", 1
        },
        {
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im4a.png)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im4b.png)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts4a.csv)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts4b.csv)", 0.3
        },
        {
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im5a.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im5b.jpg)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts5a.csv)",
            R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\pts5b.csv)", 0.75
        }
    };
    for (auto file_tuple : example_files)
    {
        Mat imgA, imgB;
        vector<vector<float>> ptsA, ptsB;

        LoadImages(get<0>(file_tuple), get<1>(file_tuple), get<2>(file_tuple), get<3>(file_tuple), get<4>(file_tuple), imgA, imgB, ptsA, ptsB);

        Mat ptsA_Mat = convertVectorOfVectorsToMat(ptsA);
        Mat ptsB_Mat = convertVectorOfVectorsToMat(ptsB);
        //cout << ptsA_Mat.at<float>(35, 0) << "," << ptsA_Mat.at<float>(35, 1);
        auto ptsAVec = convertToVectorOfPoints(ptsA);
        auto ptsBVec = convertToVectorOfPoints(ptsB);
        Mat mask;
        //Mat ret = cv::findFundamentalMat(ptsA_Mat, ptsB_Mat);
        cv::separableFundamentalMatrix::findSeparableFundamentalMat(ptsA_Mat, ptsB_Mat, imgA.size().height, imgA.size().width);

        ShowImage(imgA);
    }
    /*

    cv::String img1_name = R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1a.jpg)";
    cv::String img2_name = R"(C:\Users\AMIR\Downloads\Separable_Fundemental_matrix\images\im1b.jpg)";
    auto t = LoadImages(img1_name, img2_name, 0.5);
    ShowImage(get<0>(t));
    ShowImage(get<1>(t));*/
    
    return 0;
}

