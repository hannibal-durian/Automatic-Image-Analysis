//
// Do not change this file.
//
#ifndef CVTLIB_H
#define CVTLIB_H

#include <iostream>
#include <opencv2/opencv.hpp>

namespace given {

// Normalizes and displays an image using the OpenCV GUI API.
// img: Image to show
// win: Name of the window
// dur: Waits for dur milliseconds.
void showImage(const cv::Mat& img, std::string win="Image", double dur=0);

// This function colors a region of an image.
// img: Image to color
// dst: Colored output image
// region: Region to color.
// color: The color to color the region with.
void colorImage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& region, const cv::Vec3b& color);

// This function is just a color map, that maps an integer number to a color.
// The colors are supposed to be easily distinctable.
// id: Integer index
cv::Vec3b getColor(int id);

// Plot fourier descriptor
//
// fd: The fourier descriptor to be displayed
// win: The window name
// dur:	Wait number of ms or until key is pressed
void plotFourierDescriptor(const cv::Mat& fd, std::string win, double dur);

}

#endif // GIVEN_H
