//============================================================================
// Author      : Ronny Haensch
// Version     : 1.1
// Copyright   : -
// Description : App that does GHT and shows results
//============================================================================
#include <string>
#include <opencv2/opencv.hpp>

#include "given.hpp"
#include "yours.hpp"

using namespace std;
using namespace cv;

void app(Mat& templateImage, Mat& queryImage, vector<Scalar>& objList, Mat& params){

    // processing parameter
    double sigma            = params.at<float>(0);      // standard deviation of directional gradient kernel
    double templateThresh   = params.at<float>(1);      // relative threshold for binarization of the template image
    double objThresh        = params.at<float>(2);      // relative threshold for maxima in hough space
    double scaleSteps       = params.at<float>(3);      // scale resolution in terms of number of scales to be investigated
    double scaleRange[2];                               // scale of angles [min, max]
    scaleRange[0]           = params.at<float>(4);
    scaleRange[1]           = params.at<float>(5);
    double angleSteps       = params.at<float>(6);      // angle resolution in terms of number of angles to be investigated
    double angleRange[2];                               // range of angles [min, max)
    angleRange[0]           = params.at<float>(7);
    angleRange[1]           = params.at<float>(8);

    // calculate directional gradient of images as complex numbers (two channel image)
    Mat gradImage = given::calcDirectionalGrad(queryImage, sigma);
    Mat gradTempl = given::calcDirectionalGrad(templateImage, sigma);

    // generate template from template image
    // templ[0] == binary image
    // templ[0] == directional gradient image
    Mat binTempl = yours::binarizeGradientImage(gradTempl, templateThresh);
    vector<Mat> templ = {binTempl, gradTempl};

    // show binary image
    // given::showImage(templ[0], "Binary part of template", 0);

    // perfrom general hough transformation
    vector< vector<Mat> > houghSpace = yours::generalHough(gradImage, templ, scaleSteps, scaleRange, angleSteps, angleRange);

    // plot hough space (max over angle- and scale-dimension)
    Mat hough = yours::visualizeHoughSpace(houghSpace);
    // given::showImage(hough, "hough space", 0);
    imwrite("hough_space.png", hough);


    // find maxima in hough space
    given::findHoughMaxima(houghSpace, objThresh, objList);

    // plot final detection result
    Mat result = given::visualizeResult(templ[0], queryImage, objList, scaleSteps, scaleRange, angleSteps, angleRange);
    // showImage(result, "result", 0);
    imwrite("result.png", result);

}
