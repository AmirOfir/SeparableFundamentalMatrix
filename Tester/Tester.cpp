
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include "SFM_finder.hpp"
#include "fm_finder.hpp"

using namespace cv;
using namespace cv::separableFundamentalMatrix;
using namespace std;

string shape(const cv::Mat &mat)
{
    std::ostringstream stream;
    stream << "(" << mat.size().height << "," << mat.size().width << "," << (mat.dims + 1) << ")";
    return stream.str();
}

vector<vector<double>> parseCSV(string csv_name)
{
    ifstream data(csv_name);
    string line;
    std::string::size_type sz;
    vector<vector<double>> parsedCsv;
    while(std::getline(data,line))
    {
        stringstream lineStream(line);
        string cell;
        vector<double> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(stod(cell, &sz));
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
vector<Point2d> convertToVectorOfPoints(const vector<vector<RealType>> &vec, int chs = 1)
{
    vector<Point2d> ret;
    for (auto c : vec)
        ret.push_back(Point2d(c.at(0), c.at(1)));
    return ret;
}

struct ExampleFile
{
    string img1_name;
    string img2_name; 
    string imgA_pts_name; 
    string imgB_pts_name;
    float scale_factor;
    ExampleFile(string baseFolder, string _img1_name, string _img2_name, string _imgA_pts_name, string _imgB_pts_name, float _scale_factor)
        : img1_name(baseFolder + _img1_name), img2_name(baseFolder + _img2_name), 
        imgA_pts_name(baseFolder + _imgA_pts_name), imgB_pts_name(baseFolder + _imgB_pts_name), scale_factor(_scale_factor)
    { }
};


void LoadImages(ExampleFile exampleFile, 
    Mat &img1, Mat &img2, vector<vector<double>> &ptsA, vector<vector<double>> &ptsB)
{
    //string img1_name, string img2_name, string imgA_pts_name, string imgB_pts_name, float scale_factor
    auto img1c = imread(exampleFile.img1_name); // queryImage
    auto img2c = imread(exampleFile.img2_name); // trainImage
    
    if (exampleFile.scale_factor < 1)
    {
        Mat tmp;
        std::cout << "Image shape before scaling " << shape(img1c) << endl;
        resize(img1c, tmp, Size(), exampleFile.scale_factor, exampleFile.scale_factor, cv::INTER_AREA);
        img1c = tmp;

        resize(img2c, tmp, Size(), exampleFile.scale_factor, exampleFile.scale_factor, cv::INTER_AREA);
        img2c = tmp;
    }
    std::cout << "Image shape after scaling " << shape(img1c) << endl;

    cv::cvtColor(img1c, img1c, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2c, img2c, cv::COLOR_BGR2GRAY);

    ptsA = parseCSV(exampleFile.imgA_pts_name);
    ptsB = parseCSV(exampleFile.imgB_pts_name);
    img1 = img1c;
    img2 = img2c;
}


int main(int argc, char *argv[])
{
    std::string argv_str(argv[0]);
    std::string base;
    if (argv_str.find_last_of("/") == -1)
        base = argv_str.substr(0, argv_str.find_last_of("\\") + 1);
    else
        base = argv_str.substr(0, argv_str.find_last_of("/") + 1);

    vector<ExampleFile> exampleFiles = 
    {
        {
            base, R"(im1a.jpg)", R"(im1b.jpg)", R"(pts1a.csv)", R"(pts1b.csv)", 0.5
        },
        {
            base, R"(im2a.jpg)", R"(im2b.jpg)", R"(pts2a.csv)", R"(pts2b.csv)", 1
        },
        {
            base, R"(im3a.jpg)", R"(im3b.jpg)", R"(pts3a.csv)", R"(pts3b.csv)", 1
        },
        {
            base, R"(im4a.png)", R"(im4b.png)", R"(pts4a.csv)", R"(pts4b.csv)", 0.3
        },
        {
            base, R"(im5a.jpg)", R"(im5b.jpg)", R"(pts5a.csv)", R"(pts5b.csv)", 0.75
        }
    };

    for (auto exampleFile : exampleFiles)
    {
        Mat imgA, imgB;
        vector<vector<double>> ptsA, ptsB;

        LoadImages(exampleFile, imgA, imgB, ptsA, ptsB);

        Mat ptsA_Mat = convertVectorOfVectorsToMat(ptsA);
        Mat ptsB_Mat = convertVectorOfVectorsToMat(ptsB);
        //cout << ptsA_Mat.at<float>(35, 0) << "," << ptsA_Mat.at<float>(35, 1);
        auto ptsAVec = convertToVectorOfPoints(ptsA);
        auto ptsBVec = convertToVectorOfPoints(ptsB);
        Mat mask;
        
        Mat retCV = cv::findFundamentalMat(ptsA_Mat, ptsB_Mat, mask, FM_RANSAC);
        std::cout << retCV << endl;

        retCV = cv::separableFundamentalMatrix::findFundamentalMatFullRansac(ptsA_Mat, ptsB_Mat);
        std::cout << retCV << endl;

        cv::separableFundamentalMatrix::SeparableFundamentalMatFindCommand command
            (ptsA_Mat, ptsB_Mat, imgA.size().height, imgA.size().width, 0.4, 3, -1, 150, 4, 4, 180, 3, 2, 4);

        auto topMatchingLines = command.FindMatchingLines();
        
        int bestInliersCount = 0;
        Mat bestInliersMat;
        for (auto topLine: topMatchingLines)
        {
            Mat m;
            int i;
            if (command.FindMat(topLine, m, i))
            {
                if (i > bestInliersCount)
                {
                    bestInliersCount = i;
                    bestInliersMat = m;
                }
            }
        }

    
        bestInliersMat = command.TransformResultMat(bestInliersMat);

        
        Mat ret = cv::separableFundamentalMatrix::findSeparableFundamentalMat(ptsA_Mat, ptsB_Mat, imgA.size().height, imgA.size().width);
        cout << "Mat:" << endl << ret << endl;

    }
    
    return 0;
}