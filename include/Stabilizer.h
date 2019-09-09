#pragma once

#include <opencv2/opencv.hpp>

#include "MeshFlow.h"
#include "Optimization.h"

#define Log(x) std::cout<<x<<std::endl

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

    void stabilize();

    void getFrameWarp();

    void generateStabilizedVideo(VideoCapture& cap);

private:
    std::vector<cv::Mat> smoothXPaths, smoothYPaths;
    std::vector<cv::Mat> xPaths, yPaths;
    std::vector<cv::Mat> xMotionMeshes, yMotionMeshes;
    std::vector<cv::Mat> newXMotionMeshes, newYMotionMeshes;

};

}
