//============================================================================
// Author      : Ronny Haensch
// Version     : 1.1
// Copyright   : -
// Description : loads images and calls app
//============================================================================
#include <iostream>

#include <opencv2/opencv.hpp>

#include "app.cpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    // check if image paths were defined
    if ( argc != 3 ) {
        cerr << "Usage: app <path to template image> <path to query image>" << endl;
        return -1;
    }

    // load template image 
    Mat templateImage = imread(argv[1], IMREAD_GRAYSCALE);
    if (!templateImage.data){
        cerr << "ERROR: Cannot load template image from\n" << argv[1] << endl;
        exit(-1);
    }
    // convert 8U to 32F
    templateImage.convertTo(templateImage, CV_32F);
    // show template image
    given::showImage(templateImage, "Template image", 0);

    // load test image
    Mat queryImage = imread(argv[2], IMREAD_GRAYSCALE);
    if (!queryImage.data){
        cerr << "ERROR: Cannot load query image from\n" << argv[2] << endl;
        exit(-1);
    }
    // and convert it from 8U to 32F
    queryImage.convertTo(queryImage, CV_32F);
    // show test image
    given::showImage(queryImage, "queryImage", 0);

    // define parameters ------------------------------------------------------
    double sigma            = 0.9;        // standard deviation of directional gradient kernel
    double templateThresh   = 0.7;      // relative threshold for binarization of the template image
    double objThresh        = 0.9;        // relative threshold for maxima in hough space
    double scaleSteps       = 5;        // scale resolution in terms of number of scales to be investigated
    double scaleRange[2];               // scale of angles [min, max]
    scaleRange[0]           = 1;
    scaleRange[1]           = 5;
    double angleSteps       = 12;        // angle resolution in terms of number of angles to be investigated
    double angleRange[2];               // range of angles [min, max)
    angleRange[0]           = 0;
    angleRange[1]           = 2*CV_PI;

    Mat params = (Mat_<float>(1,9) << sigma, templateThresh, objThresh, scaleSteps, scaleRange[0], scaleRange[1], angleSteps, angleRange[0], angleRange[1]);
    // ------------------------------------------------------------------------

    // start processing
    vector<Scalar> objList;
    app(templateImage, queryImage, objList, params);

    // print found objects on screen
    // cout << "Number of objects: " << objList.size() << endl;
    // int i=0;
    // for(vector<Scalar>::iterator it = objList.begin(); it != objList.end(); it++, i++){
    //     cout << i << "\tScale:\t" << (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
    //     cout << "\tAngle:\t" << ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0])/CV_PI*180;
    //     cout << "\tPosition:\t(" << (*it).val[2] << ", " << (*it).val[3] << " )" << endl;
    // }

    return 0;
}
