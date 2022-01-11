//
// Do not change this file.
//
#include "given.hpp"
#include <opencv2/opencv.hpp>

void given::smoothImage(const cv::Mat& src, cv::Mat& dst, int size) {
    cv::GaussianBlur(src, dst, cv::Size(size, size), 0, 0);
}

void given::showImage(const cv::Mat& img, std::string win, double dur) {
    // use copy for normalization
    cv::Mat copy = img.clone();
    if (img.channels() == 1) cv::normalize(img, copy, 0, 255, cv::NORM_MINMAX);
    // create window and display image
    cv::namedWindow(win.c_str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(win.c_str(), copy);
    // wait
    if (dur >= 0) cv::waitKey(dur);
}
