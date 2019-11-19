#ifndef SYSTEM_H
#define SYSTEM_H

#include <exception>
#include <opencv2/core/core.hpp>

#include "MeshFlow.h"
#include "Optimization.h"
#include "Configurations.h"

#define Log(x) std::cout<<x<<std::endl

namespace meshflow {

class System
{
public:
    System(std::string videoFilename, std::string configFilename = "");

    void readVideo(cv::VideoCapture& cap);

    void optimize(bool online);

    void stabilize();

    void getFrameWarp();

    void generateStabilizedVideo(cv::VideoCapture& cap);

private:
    cv::Size cellSize;
    int HORIZONTAL_BORDER, VERTICAL_BORDER;
    std::string videoFilename;
    cv::Size frameResolution;
    cv::VideoCapture videoIn;

    config::MainConfiguration mainConfig;

    std::vector<cv::Mat> smoothXPaths, smoothYPaths;
    std::vector<cv::Mat> xPaths, yPaths;
    std::vector<cv::Mat> xMotionMeshes, yMotionMeshes;
    std::vector<cv::Mat> newXMotionMeshes, newYMotionMeshes;

    std::vector<std::pair<double, double>> lambdas;

    double width, height, cellHeight, cellWidth;

};

}

#endif
