#include <iostream>
#include <vector>

#include "given.hpp"
#include "yours.hpp"

struct input {
    std::string path;
    cv::Mat img;
    int bin_thresh;
    int n_erosions;
};


cv::Mat app(std::vector<input> templates, input query, int descriptor_length, float detection_thresh) {
    // temporary images and copy of query image to color the result -----------
    cv::Mat temp;
    cv::Mat result = query.img.clone();

    // get contour lines from templates ---------------------------------------
    std::vector<cv::Mat> template_contours;
    for (auto& t : templates) {
        std::vector<cv::Mat> contours;
        yours::preprocessImage(t.img, temp, t.bin_thresh, t.n_erosions);
        cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // take only the contour with the most points
        int max_size = -1;
        cv::Mat contour;
        for (auto const& c : contours) {
            if (c.rows > max_size) {
                max_size = c.rows;
                contour = c;
            }
        }

        // stop execution if no contour with sufficient length is found for template
        if (max_size < descriptor_length) {
            std::cout << "No contour of length " << std::to_string(descriptor_length) << " found for " << t.path << std::endl;
            exit(-1);
        }

        template_contours.push_back(contour);
    }

    // get contour lines from query image -------------------------------------
    std::vector<cv::Mat> query_contours;
    yours::preprocessImage(query.img, temp, query.bin_thresh, query.n_erosions);
    cv::findContours(temp, query_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // skip those that have not enough elements and color them white
    for (auto it = query_contours.begin(); it != query_contours.end();) {
        if (it->rows < descriptor_length) {
            given::colorImage(result, result, *it, cv::Vec3b(255, 255, 255));
            it = query_contours.erase(it);
        } else {
             ++it;
        }
    }

    // stop execution if no contour with sufficient length is found in query image
    if (query_contours.size() <= 0) {
        std::cout << "No contour of length " << std::to_string(descriptor_length) << " found for " << query.path << std::endl;
        exit(-1);
    }

    // get fourier descriptors ------------------------------------------------
    std::vector<cv::Mat> template_fds;
    for (auto const& c : template_contours)
        template_fds.push_back(yours::getFourierDescriptor(c));

    std::vector<cv::Mat> query_fds;
    for (auto const& c : query_contours)
        query_fds.push_back(yours::getFourierDescriptor(c));

    // normalize fourier descriptors ------------------------------------------
    std::vector<cv::Mat> template_fds_normalized;
    for (auto const& fd : template_fds)
        template_fds_normalized.push_back(yours::normalizeFourierDescriptor(fd, descriptor_length));

    std::vector<cv::Mat> query_fds_normalized;
    for (auto const& fd : query_fds)
        query_fds_normalized.push_back(yours::normalizeFourierDescriptor(fd, descriptor_length));

    // classify objects -------------------------------------------------------
    std::vector<int> classes;
    for (auto const& fd : query_fds_normalized)
        classes.push_back(yours::classifyFourierDescriptor(fd, template_fds_normalized, detection_thresh));

    // color result image -----------------------------------------------------
    for (int i = 0; i < classes.size(); i++) {
        if (classes[i] < 0) {  // color red if not classified
            given::colorImage(result, result, query_contours[i], cv::Vec3b(0, 0, 255));
        } else {
            given::colorImage(result, result, query_contours[i], given::getColor(classes[i]));
        }
    }

    // save to disk -----------------------------------------------------------
    // given::showImage(result, "result", 0);
    std::string path = query.path.replace(query.path.length()-4, 4, "_result.jpg");
    std::cout << "Saving result to " << path << std::endl;
    cv::imwrite(path, result);

    // return semantic segmentation -------------------------------------------
    cv::Mat segmentation(query.img.rows, query.img.cols, CV_8UC3, 0.0);
    for (int i = 0; i < query_contours.size(); i++)
        if (classes[i] >= 0)
            cv::drawContours(segmentation, query_contours, i, given::getColor(classes[i]), -1);

    return segmentation;
}
