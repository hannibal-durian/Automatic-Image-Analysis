//
// Do not change this file.
//
#ifndef YOURS_H
#define YOURS_H

#include <iostream>
#include <opencv2/opencv.hpp>


namespace yours {

// TODO(student):
// Converts input image to gray scale and binarizes image.
// src: Input image
// dst: Destination for binarized image
// thresh: Threshold for binarization
void binarizeImage(const cv::Mat& src, cv::Mat& dstm, int thresh=125);

// TODO(student):
// Smoothes and binarizes an image.
// image: Input image.
//
// returns: Processed image.
cv::Mat processImage(const cv::Mat& image);

}

#endif // YOURS_H
