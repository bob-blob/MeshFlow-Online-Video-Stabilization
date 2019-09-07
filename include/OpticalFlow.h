#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;
using cv::Point2f;
using std::vector;

namespace meshflow {

class FastKltTracker {
public:
    void getMatches(const Mat& source, const Mat& target,
                          vector<Point2f>& sourceFeatures, vector<Point2f>& targetFeatures);

private:

};


}
