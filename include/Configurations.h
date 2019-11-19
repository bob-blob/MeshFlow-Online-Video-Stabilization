#ifndef CONFIGURATIONS_H
#define CONFIGURATIONS_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>

namespace meshflow {

namespace config {

enum class FeatureExtractionMethod { FAST, SHI_TOMASI, UNDEFINED };

class FeatureExtractionParameters {};

struct FASTParameters {
    int threshold = 10;
    bool nonmaxSuppression = true;
    int type = cv::FastFeatureDetector::TYPE_9_16;
};

struct GoodFeaturesToTrackParameters {
    int maxCorners = 1000;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;
    bool useHarris = false;
    double k= 0.04;
};

enum class CornerSubPixState { ENABLED, DISABLED, UNDEFINED };

struct CornerSubPixParameters {
    cv::Size winSize = cv::Size(15, 15);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria termCriteria = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
};

struct KLTTrackerParameters {
    cv::Size trackingGridSize = cv::Size(4, 4);
    cv::Size winSize = cv::Size(15, 15);
    int maxPyramidLevel = 2;
    cv::TermCriteria termCriteria = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
};

enum class OutlierRejectionState { SIMPLE, HOMOGRAPHY_FIT, UNDEFINED };
enum class FittingStrategy { STANDARD, RANSAC, LMEDS };

struct HomographyFit {
    FittingStrategy fitStrategy = FittingStrategy::RANSAC;
    double rejectionThreshold = 2;
};

struct OutlierRejectionParameters {
    OutlierRejectionState state = OutlierRejectionState::HOMOGRAPHY_FIT;
    cv::Size gridSize = cv::Size(4, 4);
    HomographyFit homographyFit;
};

struct OpticalFlowConfiguration {
    FeatureExtractionMethod extractionMethod = FeatureExtractionMethod::SHI_TOMASI;
    FASTParameters fastParams;
    GoodFeaturesToTrackParameters goodFeaturesParams;
    CornerSubPixState subpixState = CornerSubPixState::ENABLED;
    CornerSubPixParameters subpixParams;
    KLTTrackerParameters kltParams;
    OutlierRejectionParameters outlierParams;
};

enum class OptimizationWeightStrategy {
    CONSTANT_WEIGHTING,
    ADAPTIVE_WEIGHTING,
    UNDEFINED
};

enum class OptimizationMode {
    OFFLINE,
    ONLINE,
    UNDEFINED
};

enum class ApplicationMode {
    SINGLE_THREADED,
    MULTI_THREADED,
    UNDEFINED
};

struct OptimizationConfiguration {
    OptimizationMode mode = OptimizationMode::OFFLINE;
    OptimizationWeightStrategy weightStrategy = OptimizationWeightStrategy::CONSTANT_WEIGHTING;
    int iterations = 100;
    int windowSize = 6;     // temporal window
    // ONLINE_PARAMS
    int frameBufferSize = 60;
    int beta = 1;
};

struct MeshFlowConfiguration {
    cv::Size gridSize = cv::Size(16, 16); // (cols, rows)
    int motionPropagationRadius = 300;
};

struct ApplicationConfiguration {
    ApplicationMode mode = ApplicationMode::SINGLE_THREADED;
};

struct MainConfiguration {
    ApplicationConfiguration appConfig;
    OpticalFlowConfiguration optFlowConfig;
    MeshFlowConfiguration meshFlowConfig;
    OptimizationConfiguration optimConfig;
};

// Find a way to generalize this ...
OptimizationMode getOptimizationMode(std::string mode);
OptimizationWeightStrategy getOptimizationWeightingStrategy(std::string mode);
FeatureExtractionMethod getFeatureExtractionMethod(std::string method);
CornerSubPixState getCornerSubPixState(std::string state);
OutlierRejectionState getOutlierRejectionState(std::string state);
FittingStrategy getFittingStrategy(std::string mode);

std::string getOptimizationMode_string(OptimizationMode mode);
std::string getOptimizationWeightingStrategy_string(OptimizationWeightStrategy mode);
std::string getCornerSubPixState_string(CornerSubPixState state);
std::string getFeatureExtractionMethod_string(FeatureExtractionMethod method);
std::string getOutlierRejectionState_string(OutlierRejectionState state);
std::string getFittingStrategy_string(FittingStrategy mode);

MainConfiguration loadConfigs(std::string filename);
OpticalFlowConfiguration loadOpticalFlowConfig(const cv::FileStorage& fs);
MeshFlowConfiguration loadMeshFlowConfig(const cv::FileStorage& fs);
OptimizationConfiguration loadOptimizationConfig(const cv::FileStorage& fs);

void saveConfigs(std::string filename);

}   // namespace config

}   // namespace meshflow

#endif // CONFIGURATIONS_H
