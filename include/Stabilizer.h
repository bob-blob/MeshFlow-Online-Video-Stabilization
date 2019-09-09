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

    void readVideo(VideoCapture& cap,
                   std::vector<Mat>& xMotionMeshes, std::vector<Mat>& yMotionMeshes,
                   std::vector<Mat>& xPaths, std::vector<Mat>& yPaths);

    void stabilize(const std::vector<Mat>& xPaths, const std::vector<Mat>& yPaths,
                   std::vector<Mat>& smoothXPaths, std::vector<Mat>& smoothYPaths);

    void getFrameWarp(const std::vector<Mat>& xPaths, const std::vector<Mat>& yPaths,
                      const std::vector<Mat>& smoothXPaths, const std::vector<Mat>& smoothYPaths,
                      std::vector<Mat>& xMotionMeshes, std::vector<Mat>& yMotionMeshes,
                      std::vector<Mat>& newXMotionMeshes, std::vector<Mat>& newYMotionMeshes);

    void generateStabilizedVideo(VideoCapture& cap, const std::vector<Mat>& xMotionMeshes, const std::vector<Mat>& yMotionMeshes,
                                 const std::vector<Mat>& newXMotionMeshes, const std::vector<Mat>& newYMotionMeshes);
};

}
