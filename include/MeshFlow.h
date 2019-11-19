#ifndef MESHFLOW_H
#define MESHFLOW_H

#include <opencv2/core/core.hpp>
#include <map>

#include "Configurations.h"

#define Log(x) std::cout<<x<<std::endl

namespace meshflow {

typedef std::map<std::pair<int, int>, std::vector<double>> Map;
static int RADIUS_X;
static int RADIUS_Y;

class MeshFlow
{
public:
    MeshFlow();
    MeshFlow(const config::MeshFlowConfiguration& config);

    void motionPropagate(
            const std::vector<cv::Point2d>& oldPoints, const std::vector<cv::Point2d>& newPoints, const cv::Mat& oldFrame,
            cv::Mat& xMotionMesh, cv::Mat& yMotionMesh, std::vector<std::pair<double, double>>& lambdas, cv::Size gridSize);
    void generateVertexProfiles(
            const cv::Mat& xMotionMesh, const cv::Mat& yMotionMesh, std::vector<cv::Mat>& xPaths, std::vector<cv::Mat>& yPaths);
    void meshWarpFrame(
            const cv::Mat& frame, const cv::Mat& xMotionMesh, const cv::Mat& yMotionMesh, cv::Mat& warpedFrame, cv::Size sz);
private:
    cv::Point2d transformPoint(const cv::Mat& H, const cv::Point2d& point);
    void initializeMotionMesh(
            const cv::Mat& homography, cv::Size meshSize, cv::Size gridCellSize, Map& xMotion, Map& yMotion);
    void propogateFeatureMotion(
            const std::vector<cv::Point2d>& oldFeatures, const std::vector<cv::Point2d>& newFeatures, const cv::Mat& homography,
            cv::Size meshSize, cv::Size gridCellSize, Map xMotion, Map& yMotion);

    void sparseMedianFilter(Map& initialMotionField, Map& propagatedMotionField, cv::Mat& filteredMotionField) ;
    void spatialMedianFilter(cv::Mat& motionField, int kernelSize);
    double translationalElement(const cv::Mat& homography, cv::Size frameResolution);
    double affineComponent(const cv::Mat& homography);

private:
    int motionPropagationRadius = 500;

};

}   // namespace meshflow

#endif
