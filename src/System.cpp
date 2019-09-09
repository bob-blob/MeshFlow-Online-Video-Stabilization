#include "System.h"

namespace meshflow {

System::System(const std::string& filename) : filename{filename} {}

void System::run() {
    cv::VideoCapture cap(filename);

    // Propagate motion vectors and generate vertex profiles
    Stabilizer stab;
    std::vector<cv::Mat> xMotionMeshes, yMotionMeshes,
                         xPaths, yPaths;
    stab.readVideo(cap, xMotionMeshes, yMotionMeshes, xPaths, yPaths);

    // Stabilize the  vertex profiles
    std::vector<cv::Mat> smoothXPaths = xPaths, smoothYPaths = yPaths;
    stab.stabilize(xPaths, yPaths, smoothXPaths, smoothYPaths);

    // Get updated mesh warps
    std::vector<cv::Mat> newXMotionMeshes, newYMotionMeshes;
    stab.getFrameWarp(xPaths, yPaths, smoothXPaths, smoothYPaths, xMotionMeshes, yMotionMeshes, newXMotionMeshes, newYMotionMeshes);

    // Apply updated mesh warps & save the results
    stab.generateStabilizedVideo(cap, xMotionMeshes, yMotionMeshes, newXMotionMeshes, newYMotionMeshes);
}

}
