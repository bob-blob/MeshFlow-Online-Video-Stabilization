#pragma once

#include <opencv2/opencv.hpp>

#include "MeshFlow.h"
#include "Optimization.h"

#define Log(x) std::cout<<x<<std::endl

using cv::Mat;
using cv::VideoCapture;
using cv::Point2f;

namespace meshflow {

/// For now implement video reading and optical flow here
class Stabilizer
{
public:
    Stabilizer(const std::string& filename, cv::Size sz) : filename(filename),
                                                           PIXELS_X(sz.width / 16),
                                                           PIXELS_Y(sz.height / 16) {};

    const int PIXELS = 16;
    int PIXELS_X;
    int PIXELS_Y;

    const int RADIUS = 300;

    int HORIZONTAL_BORDER;
    int VERTICAL_BORDER;

    void readVideo(VideoCapture& cap);

    void stabilize(bool online);

    void getFrameWarp();

    void generateStabilizedVideo(VideoCapture& cap);

    void extractFeaturesFromGrid(const Mat& frame, std::vector<cv::Point2f>& corners);

    void simpleRejection(const vector<uchar>& status, const vector<Point2f>& frameCorners, const vector<Point2f>& newFrameCorners,
                         vector<Point2d>& old_points, vector<Point2d>& new_points);

    void outlierRejection(const vector<uchar>& status, const vector<Point2f>& frameCorners, const vector<Point2f>& newFrameCorners,
                          vector<Point2d>& old_points, vector<Point2d>& new_points);
private:
    std::string filename;

    std::vector<cv::Mat> smoothXPaths, smoothYPaths;
    std::vector<cv::Mat> xPaths, yPaths;
    std::vector<cv::Mat> xMotionMeshes, yMotionMeshes;
    std::vector<cv::Mat> newXMotionMeshes, newYMotionMeshes;

    std::vector<std::pair<double, double>> lambdas;

    double width, height, cellHeight, cellWidth;

};

}
