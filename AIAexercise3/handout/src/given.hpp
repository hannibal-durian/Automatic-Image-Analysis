#ifndef __GIVEN_H_
#define __GIVEN_H_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace given {

// Normalizes and displays an image using the OpenCV GUI API.
// img: Image to show
// win: Name of the window
// dur: Waits for dur milliseconds.
void showImage(const cv::Mat& img, std::string win="Image", double dur=0);


// computes directional gradients
/*
  image:    the input image
  sigma:    standard deviation of the kernel
  return:   the two-channel gradient image
*/
Mat calcDirectionalGrad(Mat& image, double sigma);


// rotates and scales a given image
/*
  image:    the image to be scaled and rotated
  angle:    rotation angle in radians
  scale:    scaling factor
  return:   transformed image
*/
Mat rotateAndScale(Mat& image, double angle, double scale);


// shows the detection result of the hough transformation
/*
  testImage:    the test image, where objects were searched (and hopefully found)
  templ:        the template consisting of binary image and complex-valued directional gradient image
  objList:      list of objects as defined by findHoughMaxima(..)
  scaleSteps:   scale resolution
  scaleRange:   range of investigated scales [min, max]
  angleSteps:   angle resolution
  angleRange:   range of investigated angles [min, max)
*/
Mat visualizeResult(Mat& templateImage, Mat& queryImage, vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);


// seeks for local maxima within the hough space
/*
  a local maxima has to be larger than all its 8 spatial neighbors, as well as the largest value at this position for all scales and orientations
  houghSpace:   the computed hough space
  objThresh:    relative threshold for maxima in hough space
  objList:  list of detected objects
*/
void findHoughMaxima(vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList);


// Performes a circular shift in (dx,dy) direction
/*
in:         input matrix
out:    circular shifted matrix
dx:         shift in x-direction
dy:         shift in y-direction
*/
void circShift(Mat& in, Mat& out, int dx, int dy);

}

#endif // __GIVEN_H_
