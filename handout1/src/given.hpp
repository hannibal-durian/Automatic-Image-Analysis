//
// Do not change this file.
//
#ifndef GIVEN_H
#define GIVEN_H

#include <iostream>
#include <opencv2/opencv.hpp>

namespace given {

// Smoothes an image using Gaussian blur. Sigma is derived from size.
// This wrapper is trivial and its sole purpose is to pass butter.
// src: Input image
// dst: Output image
// size: kernel size
void smoothImage(const cv::Mat& src, cv::Mat& dst, int size=3);

// Normalizes and displays an image using the OpenCV GUI API.
// img: Image to show
// win: Name of the window
// dur: Waits for dur seconds.
void showImage(const cv::Mat& img, std::string win, double dur);

}

#endif // GIVEN_H
