#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;
using cv::VideoCapture;

namespace meshflow {

/// For now implement video reading and optical flow here
class Stabilizer
{
public:
    const int PIXELS = 16;
    const int RADIUS = 300;

    int HORIZONTAL_BORDER;
    int VERTICAL_BORDER;

    void readVideo(VideoCapture& cap);

    void stabilize(const Mat& xPaths, const Mat& yPaths,
                   Mat& smoothXPaths, Mat& smoothYPaths);
};

}
