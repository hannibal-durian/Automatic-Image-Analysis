//
// Do not change this file.
//
#include "catch.hpp"
#include "../src/yours.hpp"

using namespace std;
using namespace cv;

TEST_CASE("Test preprocessImage", "preprocessImage") {
    cv::Mat inputImage, outputImage;
    inputImage = cv::imread("./data/shape.png", cv::IMREAD_COLOR);
    yours::preprocessImage(inputImage, outputImage, 125, 1);

    SECTION("Channels") {
        INFO("There is a problem with the channels of the output image!");
        REQUIRE(outputImage.channels() == 1);
    }

    SECTION("Size") {
        INFO("There is a problem with the size of the output image!");
        REQUIRE(outputImage.cols == inputImage.cols);
        REQUIRE(outputImage.rows == inputImage.rows);
    }
}

TEST_CASE("Test getFourierDescriptor", "getFourierDescriptor") {
    Mat cline(68, 1, CV_32SC2);
    int k = 0;
    for (int i = 41; i < 58; i++) cline.at<Vec2i>(k++) = Vec2i(41, i);
    for (int i = 41; i < 58; i++) cline.at<Vec2i>(k++) = Vec2i(i, 58);
    for (int i = 58; i > 41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
    for (int i = 58; i > 41; i--) cline.at<Vec2i>(k++) = Vec2i(i, 41);

    Mat fd = yours::getFourierDescriptor(cline);

    SECTION("Length") {
        INFO(
            "There is a problem with getFourierDescriptor(..): "
            "The number of frequencies does not match the number of contour points");
        REQUIRE(fd.rows == cline.rows);
    }

    SECTION("Shape") {
        INFO(
            "There is a problem with getFourierDescriptor(..): "
            "The fourier descriptor is supposed to be a two-channel, 1D matrix");
        REQUIRE(fd.channels() == 2);
    }
}

TEST_CASE("Test normalizeFourierDescriptor", "normalizeFourierDescriptor") {
    double eps = pow(10, -3);

    Mat cline(68, 1, CV_32SC2);
    int k = 0;
    for (int i = 41; i < 58; i++) cline.at<Vec2i>(k++) = Vec2i(41, i);
    for (int i = 41; i < 58; i++) cline.at<Vec2i>(k++) = Vec2i(i, 58);
    for (int i = 58; i > 41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
    for (int i = 58; i > 41; i--) cline.at<Vec2i>(k++) = Vec2i(i, 41);

    Mat fd = yours::getFourierDescriptor(cline);
    Mat nfd = yours::normalizeFourierDescriptor(fd, 32);

    SECTION("Shape") {
        INFO(
            "There is a problem with normalizeFourierDescriptor(..): "
            "The normalized fourier descriptor is supposed to be a one-channel, 1D matrix"
            " and the number of components must match the specified number of components.");
        REQUIRE(nfd.channels() == 1);
        REQUIRE(nfd.rows == 32);
    }

    SECTION("Type") {
        INFO(
             "There is be a problem with normalizeFourierDescriptor(..):"
             "The normalized fourier descriptor is supposed to be in floating point precision");
        REQUIRE(nfd.type() == CV_32FC1);
    }

    SECTION("Translation invariance") {
        INFO(
            "There is a problem with normalizeFourierDescriptor(..): "
            "The F(0)-component of the normalized fourier descriptor F is supposed to be 0.");
        REQUIRE(abs(nfd.at<float>(0)) <= eps);
    }

    SECTION("Scale invariance") {
        INFO(
            "There is a problem with normalizeFourierDescriptor(..): "
            "The F(1)-component of the normalized fourier descriptor F is supposed to be 1. "
            "But what if the unnormalized F(1)=0?");
        REQUIRE(((abs(nfd.at<float>(1) - 1.) <= eps) || (abs(nfd.at<float>(nfd.rows - 1) - 1.) <= eps)));
    }
}


TEST_CASE("Test classifyFourierDescriptor", "classifyFourierDescriptor") {

    Mat d1 = Mat(12, 1, CV_32F, {0.0, 0.0, 0.0});
    Mat d2 = Mat(12, 1, CV_32F, {0.0, 1.0, 0.0});
    Mat d3 = Mat(12, 1, CV_32F, {1.0, 2.0, 3.0});

    int r0 = yours::classifyFourierDescriptor(d1, {d3}, 0.0);
    int r1 = yours::classifyFourierDescriptor(d1, {d1, d2, d3}, 1.0);
    int r2 = yours::classifyFourierDescriptor(d1, {d2, d3}, 1.0);

    SECTION("Distance threshold") {
        INFO(
            "There is a problem with classifyFourierDescriptor(..): "
            "Classification result is not -1 for distance threshold of 0.0.")
        REQUIRE(r0 == -1);
    }

    SECTION("Example 1") {
        INFO(
            "There is a problem with classifyFourierDescriptor(..): "
            "Example 1 not classified correctly.")
        REQUIRE(r1 == 0);
    }

    SECTION("Example 2") {
        INFO(
            "There is a problem with classifyFourierDescriptor(..): "
            "Example 2 not classified correctly.")
        REQUIRE(r2 == 0);
    }



}
