#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include "Configurations.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace meshflow {

namespace OpticalFlow {

typedef std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> FramePairCorners;
typedef std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> FramePairCornersd;

inline FramePairCorners makeFramePairCorners(const std::vector<cv::Point2f> prevFrameCorners, const std::vector<cv::Point2f> nextFrameCorners) {
    return std::make_pair(prevFrameCorners, nextFrameCorners);
}

inline FramePairCornersd makeFramePairCornersd(const std::vector<cv::Point2d> prevFrameCorners, const std::vector<cv::Point2d> nextFrameCorners) {
    return std::make_pair(prevFrameCorners, nextFrameCorners);
}

void extractCornersShiTomasi(
        const cv::Mat& frame,
        std::vector<cv::Point2f>& extractedCorners,
        const config::GoodFeaturesToTrackParameters& cornerExtractionParams = config::GoodFeaturesToTrackParameters(),
        config::CornerSubPixState subPixState = config::CornerSubPixState::ENABLED,
        const config::CornerSubPixParameters& cornerSubpixParams = config::CornerSubPixParameters());

void trackKLT(
        const cv::Mat& sourceGray,
        const cv::Mat& targetGray,
        std::vector<cv::Point2d>& sourceFeatures,
        std::vector<cv::Point2d>& targetFeatures,
        const config::OpticalFlowConfiguration& optFlowConfig = config::OpticalFlowConfiguration());

void extractFASTFeaturesFromGrid(
        const cv::Mat &frame,
        std::vector<cv::Point2f>& corners,
        cv::Size gridSize = cv::Size(4, 4),
        const config::FASTParameters& fastParams = config::FASTParameters());

void extractFeaturesFromGrid(
        const cv::Mat &frame,
        std::vector<cv::Point2f>& corners,
        cv::Size gridSize = cv::Size(4, 4),
        const config::GoodFeaturesToTrackParameters& cornerExtractionParams = config::GoodFeaturesToTrackParameters(),
        config::CornerSubPixState subPixState = config::CornerSubPixState::ENABLED,
        const config::CornerSubPixParameters& cornerSubpixParams = config::CornerSubPixParameters());

void outlierRejection(const cv::Mat& frame,
                      const FramePairCorners& corners,
                      const std::vector<uchar>& status,
                      FramePairCorners& filteredCorners,
                      const config::OutlierRejectionParameters& outlierParams = config::OutlierRejectionParameters());

void rejectBadCorners(const FramePairCorners& corners,
                      const std::vector<uchar>& status,
                      FramePairCorners& filteredCorners);

}   // namespace OpticalFlow

}   // namespace meshflow

#endif
