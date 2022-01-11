//
// Do not change this file.
//
#include "given.hpp"
#include <opencv2/opencv.hpp>

void given::showImage(const cv::Mat& img, std::string win, double dur) {
    // use copy for normalization
    cv::Mat copy = img.clone();
    if (img.channels() == 1) cv::normalize(img, copy, 0, 255, cv::NORM_MINMAX);
    // create window and display image
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::imshow(win, copy);
    // wait
    if (dur >= 0) cv::waitKey(dur);
}

void given::colorImage(const cv::Mat& img, cv::Mat& out, const cv::Mat& region, const cv::Vec3b& color) {
    // copy image and duplicate channel if gray scale
    if (img.channels() == 1) {
        std::vector<cv::Mat> tmp{img, img, img};
        cv::merge(tmp, out);
    } else if (!(&img == &out)) {
        cv::Mat out = img.clone();
    }

    // color
    for (int i = 0; i < region.rows; i++)
        out.at<cv::Vec3b>(region.at<cv::Vec2i>(i)[1], region.at<cv::Vec2i>(i)[0]) = color;
}

cv::Vec3b given::getColor(int id) {
    // in BGR
    std::vector<cv::Vec3b> colors{
        cv::Vec3b(255, 0, 0),      //Blue
        cv::Vec3b(0, 255, 0),      //Green
        cv::Vec3b(0, 255, 255),    //Yellow
        cv::Vec3b(255, 0, 255),    //Magenta
        cv::Vec3b(128, 128, 255),  //Pink
        cv::Vec3b(128, 128, 128),  //Gray
        cv::Vec3b(0, 0, 128),      //Brown
        cv::Vec3b(0, 128, 255),    //Orange
    };

    if (id >= colors.size()) {
        std::cout << "Well, this is embarrassing but we ran out of colors." << std::endl;
        exit(-1);
    }

    return colors.at(id);
}


// cv::Mat binarizeLabelImage(const cv::Mat& labels, cv::Vec3b label_color) {
//     cv::Mat binary = labels != label_color;
//     std::vector<cv::Mat> chans;
//     cv::split(res, chans);
//     return chans[0] + chans[1] + chans[2];
// }

// float intersectionOverUnion(const cv::Mat& target, const cv::Mat& labels) {

// }



void given::plotFourierDescriptor(const cv::Mat& fd, std::string win, double dur) {
    // get pixel coordinates from fourier descriptor
    cv::Mat coor = cv::Mat(fd.rows, 1, CV_32FC2);
    if (fd.channels() == 1) {
        cv::Mat phase = cv::Mat::zeros(fd.rows, 1, CV_32FC1);
        std::vector<cv::Mat> channels;
        channels.push_back(cv::Mat(fd.rows, 1, CV_32FC1));
        channels.push_back(cv::Mat(fd.rows, 1, CV_32FC1));
        polarToCart(fd, phase, channels[0], channels[1]);
        merge(channels, coor);
    } else {
        fd.copyTo(coor);
    }
    dft(coor, coor, cv::DFT_INVERSE + cv::DFT_SCALE);
    std::vector<cv::Mat> cart;
    split(coor, cart);

    // transform coordinates in order that no negative coordinates remain
    double minx, miny, maxx, maxy;
    minMaxLoc(cart[0], &minx, &maxx);
    minMaxLoc(cart[1], &miny, &maxy);

    double scale = 100. / cv::max(maxx - minx, maxy - miny);

    // create image for displaying
    cv::Mat tempDisplay = cv::Mat::zeros((maxy - miny) * scale + 1, (maxx - minx) * scale + 1, CV_8UC3);
    // set pixel values
    for (int i = 0; i < coor.rows; i++) {
        tempDisplay.at<cv::Vec3b>((cart[1].at<float>(0, i) - miny) * scale, (cart[0].at<float>(0, i) - minx) * scale) = cv::Vec3b(255, 255, 255);
    }

    // display
    cv::namedWindow(win.c_str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(win.c_str(), tempDisplay);

    // wait
    if (dur >= 0) cv::waitKey(dur);
}
