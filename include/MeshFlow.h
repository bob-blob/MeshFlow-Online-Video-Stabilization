#pragma once

#include <opencv2/opencv.hpp>

using cv::Vec2d;
using cv::Mat;
using cv::Point2d;

using std::vector;

namespace meshflow {

typedef std::map<std::pair<int, int>, vector<double>> Map;

class MeshFlow
{
public:
    static const int PIXELS = 16;
    static const int RADIUS = 300;

    static void transformPoint(const Mat& H, const Point2d& point, Point2d& result);

    static void motionPropagate(const vector<Point2d> oldPoints,
                         const vector<Point2d> newPoints,
                         const Mat& oldFrame,
                         Mat& xMotionMesh, Mat& yMotionMesh);

    static void generateVertexProfiles(const Mat& xMotionMesh,
                                const Mat& yMotionMesh,
                                vector<Mat>& xPaths, vector<Mat>& yPaths);


    static void meshWarpFrame(const Mat& frame,
                       const Mat& xMotionMesh,
                       const Mat& yMotionMesh,
                       Mat& warpedFrame);

};

}
