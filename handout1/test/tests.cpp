#include "catch.hpp"

#include "../src/given.hpp"
#include "../src/yours.hpp"

using namespace std;
using namespace cv;


TEST_CASE("Test binarizition", "binarization") {

    cv::Mat inputImage, outputImage, referenceImage, referenceImageRGB;
    referenceImage = cv::imread("./data/reference_result_wo_smoothing.png", cv::IMREAD_GRAYSCALE);
    referenceImageRGB = cv::imread("./data/reference_result_rgb.png", cv::IMREAD_GRAYSCALE);
    inputImage = cv::imread("./data/test.jpg", cv::IMREAD_COLOR);
    yours::binarizeImage(inputImage, outputImage, 125);

    SECTION("Channels") {
        INFO("There is a problem with the channels of the output image!");
        REQUIRE(outputImage.channels() == 1);
    }

    SECTION("Size") {
        INFO("There is a problem with the size of the output image!");
        REQUIRE(outputImage.cols == inputImage.cols);
        REQUIRE(outputImage.rows == inputImage.rows);
    }

    SECTION("Expected result") {
        INFO("Result does not match reference result!");
        REQUIRE(cv::countNonZero(outputImage != referenceImage) == 0);
    }

    SECTION("Wrong channel order result") {
        INFO("Default channel order in OpenCV is BGR!\n https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/");
        REQUIRE(cv::countNonZero(outputImage != referenceImageRGB) != 0);
    }

}


TEST_CASE("Process image", "process_image") {

    cv::Mat inputImage, outputImage, referenceImage, referenceImageSmoothing;
    referenceImage = cv::imread("./data/reference_result.png", cv::IMREAD_GRAYSCALE);
    referenceImageSmoothing = cv::imread("./data/reference_result_wo_smoothing.png", cv::IMREAD_GRAYSCALE);
    inputImage = cv::imread("./data/test.jpg", cv::IMREAD_COLOR);
    outputImage = yours::processImage(inputImage);

    SECTION("Channels") {
        INFO("There is a problem with the channels of the output image!");
        REQUIRE(outputImage.channels() == 1);
    }

    SECTION("Size") {
        INFO("There is a problem with the size of the output image!");
        REQUIRE(outputImage.cols == inputImage.cols);
        REQUIRE(outputImage.rows == inputImage.rows);
    }

    SECTION("Expected result") {
        INFO("Result does not match reference result!");
        REQUIRE(cv::countNonZero(outputImage != referenceImage) == 0);
    }

    SECTION("Smoothing") {
        INFO("Did you forget to smooth the image?");
        REQUIRE(cv::countNonZero(outputImage != referenceImageSmoothing) != 0);
    }

}
