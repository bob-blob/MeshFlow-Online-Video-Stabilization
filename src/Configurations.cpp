#include "Configurations.h"

namespace meshflow {

namespace config {

void saveConfigs(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    config::MeshFlowConfiguration meshFlowConfig;
    fs << "meshflow" << "{";
    fs << "grid_size" << meshFlowConfig.gridSize;
    fs << "motion_propagation_radius" << meshFlowConfig.motionPropagationRadius;
    fs << "}";

    config::OpticalFlowConfiguration opticalFlowConfig;
    fs << "optical_flow" << "{";
    fs << "feature_extraction_method" << getFeatureExtractionMethod_string(opticalFlowConfig.extractionMethod);
    config::FASTParameters fastParams;
    fs << "FAST_parameters" << "{";
    fs << "type" << fastParams.type;
    fs << "nonmax_suppression" << fastParams.nonmaxSuppression;
    fs << "threshold" << fastParams.threshold;
    fs << "}";  // FAST params
    config::GoodFeaturesToTrackParameters gfttParams;
    fs << "shi_tomasi_params" << "{";
    fs << "max_corners" << gfttParams.maxCorners;
    fs << "quality_level" << gfttParams.qualityLevel;
    fs << "min_distance" << gfttParams.minDistance;
    fs << "block_size" << gfttParams.blockSize;
    fs << "use_harris" << gfttParams.useHarris;
    fs << "k" << gfttParams.k;
    fs << "}";  // GFTT params
    fs << "corner_subpix_state" << getCornerSubPixState_string(opticalFlowConfig.subpixState);
    config::CornerSubPixParameters subPixParams;
    fs << "corner_subpix" << "{";
    fs << "window_size" << subPixParams.winSize;
    fs << "zero_zone" << subPixParams.zeroZone;
    cv::TermCriteria cornerSubPixTermCrit = subPixParams.termCriteria;
    fs << "term_criteria" << "{";
    fs << "type" << cornerSubPixTermCrit.type;
    fs << "max_count" << cornerSubPixTermCrit.maxCount;
    fs << "epsilon" << cornerSubPixTermCrit.epsilon;
    fs << "}";
    fs << "}";  // CornerSubPix params
    config::KLTTrackerParameters kltParams;
    fs << "klt_params" << "{";
    fs << "track_grid_size" << kltParams.trackingGridSize;
    fs << "window_size" << kltParams.winSize;
    fs << "max_pyramid_level" << kltParams.maxPyramidLevel;
    cv::TermCriteria kltTermCriteria = kltParams.termCriteria;
    fs << "term_criteria" << "{";
    fs << "type" << kltTermCriteria.type;
    fs << "max_count" << kltTermCriteria.maxCount;
    fs << "epsilon" << kltTermCriteria.epsilon;
    fs << "}";
    fs << "}";  // klt_params
    config::OutlierRejectionParameters outlierParams;
    fs << "outlier_rejection_params" << "{";
    fs << "type" << getOutlierRejectionState_string(outlierParams.state);
    fs << "grid_size" << outlierParams.gridSize;
    fs << "homography_fit" << "{";
    fs << "fitting_strategy" << getFittingStrategy_string(outlierParams.homographyFit.fitStrategy);
    fs << "rejection_threshold" << outlierParams.homographyFit.rejectionThreshold;
    fs << "}";  // homography_fit
    fs << "}";  // outlier_rejection_params
    fs << "}";  // optical_flow

    fs << "optimization" << "{";
    config::OptimizationConfiguration optimConfig;
    fs << "mode" << getOptimizationMode_string(optimConfig.mode);
    fs << "weighting_strategy" << getOptimizationWeightingStrategy_string(optimConfig.weightStrategy);
    fs << "iterations" << optimConfig.iterations;
    fs << "temporal_window_size" << optimConfig.windowSize;
    fs << "frame_buffer_size" << optimConfig.frameBufferSize;
    fs << "beta" << optimConfig.beta;
    fs << "}";  // optimization
}

MainConfiguration loadConfigs(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    MainConfiguration mainConfig;
    //mainConfig.appConfig = loadAppConfid(fs);
    mainConfig.optFlowConfig = loadOpticalFlowConfig(fs);
    mainConfig.meshFlowConfig = loadMeshFlowConfig(fs);
    mainConfig.optimConfig = loadOptimizationConfig(fs);

    return mainConfig;
}

OptimizationConfiguration loadOptimizationConfig(const cv::FileStorage& fs) {
    cv::FileNode optimization = fs["optimization"];

    OptimizationConfiguration optimConfig;
    std::string mode, weightingStrategy;
    optimization["mode"] >> mode;
    optimConfig.mode = getOptimizationMode(mode);
    optimization["weighting_strategy"] >> weightingStrategy;
    optimConfig.weightStrategy = getOptimizationWeightingStrategy(mode);
    optimConfig.iterations = static_cast<int>(optimization["iterations"]);
    optimConfig.windowSize = static_cast<int>(optimization["temporal_window_size"]);
    optimConfig.frameBufferSize = static_cast<int>(optimization["frame_buffer_size"]);
    optimConfig.beta = static_cast<int>(optimization["beta"]);

    return optimConfig;
}

MeshFlowConfiguration loadMeshFlowConfig(const cv::FileStorage& fs) {
    cv::FileNode meshFlow = fs["meshflow"];

    MeshFlowConfiguration meshFlowConfig;
    //cv::FileNode grid_size = meshFlow["grid_size"];
    meshFlow["grid_size"] >> meshFlowConfig.gridSize;//cv::Size(static_cast<int>(grid_size["cols"]), static_cast<int>(grid_size["rows"]));
    meshFlow["motion_propagation_radius"] >> meshFlowConfig.motionPropagationRadius;

    return meshFlowConfig;
}

OpticalFlowConfiguration loadOpticalFlowConfig(const cv::FileStorage& fs) {
    cv::FileNode opticalFlow = fs["optical_flow"];

    OpticalFlowConfiguration opticalFlowConfig;
    std::string method;
    opticalFlow["feature_extraction_method"] >> method;
    opticalFlowConfig.extractionMethod = getFeatureExtractionMethod(method);

    cv::FileNode fastParams = opticalFlow["FAST_parameters"];
    FASTParameters params;
    fastParams["type"] >> params.type;
    fastParams["nonmax_suppression"] >> params.nonmaxSuppression;
    fastParams["threshold"] >> params.threshold;
    opticalFlowConfig.fastParams = params;

    cv::FileNode shiTomasiParams = opticalFlow["shi_tomasi_params"];
    GoodFeaturesToTrackParameters gfttParams;
    shiTomasiParams["max_corners"] >> gfttParams.maxCorners;
    shiTomasiParams["quality_level"] >> gfttParams.qualityLevel;
    shiTomasiParams["min_distance"] >> gfttParams.minDistance;
    shiTomasiParams["block_size"] >> gfttParams.blockSize;
    shiTomasiParams["use_harris"] >> gfttParams.useHarris;
    shiTomasiParams["k"] >> gfttParams.k;
    opticalFlowConfig.goodFeaturesParams = gfttParams;

    std::string cornerSubPixState;
    opticalFlow["corner_subpix_state"] >> cornerSubPixState;
    opticalFlowConfig.subpixState = getCornerSubPixState(cornerSubPixState);

    cv::FileNode cornerSubPix = opticalFlow["corner_subpix"];
    CornerSubPixParameters subPixParams;
    cornerSubPix["window_size"] >> subPixParams.winSize;
    cornerSubPix["zero_zone"] >> subPixParams.zeroZone;

    cv::FileNode cornerSubPixTermCriteria = cornerSubPix["term_criteria"];
    int termCritType, maxCount, epsilon;
    cornerSubPixTermCriteria["type"] >> termCritType;
    cornerSubPixTermCriteria["max_count"] >> maxCount;
    cornerSubPixTermCriteria["epsilon"] >> epsilon;
    subPixParams.termCriteria = cv::TermCriteria(termCritType, maxCount, epsilon);
    opticalFlowConfig.subpixParams = subPixParams;

    cv::FileNode kltParams = opticalFlow["klt_params"];
    KLTTrackerParameters kltParameters;
    kltParams["track_grid_size"] >> kltParameters.trackingGridSize;
    kltParams["window_size"] >> kltParameters.winSize;
    kltParams["max_pyramid_level"] >> kltParameters.maxPyramidLevel;

    cv::FileNode kltTermCriteria = kltParams["term_criteria"];
    kltTermCriteria["type"] >> termCritType;
    kltTermCriteria["max_count"] >> maxCount;
    kltTermCriteria["epsilon"] >> epsilon;
    kltParameters.termCriteria = cv::TermCriteria(termCritType, maxCount, epsilon);
    opticalFlowConfig.kltParams = kltParameters;

    cv::FileNode outlierParams = opticalFlow["outlier_rejection_params"];
    OutlierRejectionParameters outlierRejectionParams;
    outlierRejectionParams.state = getOutlierRejectionState(outlierParams["type"]);
    outlierParams["grid_size"] >> outlierRejectionParams.gridSize;

    cv::FileNode outHomographyPolicy = outlierParams["homography_fit"];
    outlierRejectionParams.homographyFit.fitStrategy = getFittingStrategy(outHomographyPolicy["fitting_strategy"]);
    outHomographyPolicy["rejection_threshold"] >> outlierRejectionParams.homographyFit.rejectionThreshold;
    opticalFlowConfig.outlierParams = outlierRejectionParams;

    return opticalFlowConfig;
}

OptimizationMode getOptimizationMode(std::string mode) {
    if (mode == "offline") {
        return OptimizationMode::OFFLINE;
    } else if (mode == "online") {
        return OptimizationMode::ONLINE;
    }
    return OptimizationMode::UNDEFINED;
}

OptimizationWeightStrategy getOptimizationWeightingStrategy(std::string mode) {
    if (mode == "adaptive") {
        return OptimizationWeightStrategy::ADAPTIVE_WEIGHTING;
    } else if (mode == "constant") {
        return OptimizationWeightStrategy::CONSTANT_WEIGHTING;
    }
    return OptimizationWeightStrategy::UNDEFINED;
}

FeatureExtractionMethod getFeatureExtractionMethod(std::string method) {
    if (method == "FAST") {
        return FeatureExtractionMethod::FAST;
    } else if (method == "shi-tomasi") {
        return FeatureExtractionMethod::SHI_TOMASI;
    }
    return FeatureExtractionMethod::UNDEFINED;
}

CornerSubPixState getCornerSubPixState(std::string state) {
    if (state == "enabled") {
        return CornerSubPixState::ENABLED;
    } else if (state == "disabled") {
        return CornerSubPixState::DISABLED;
    }
    return CornerSubPixState::UNDEFINED;
}

OutlierRejectionState getOutlierRejectionState(std::string state) {
    if (state == "simple") {
        return OutlierRejectionState::SIMPLE;
    } else if (state == "homography") {
        return OutlierRejectionState::HOMOGRAPHY_FIT;
    }
    return OutlierRejectionState::UNDEFINED;
}

FittingStrategy getFittingStrategy(std::string mode) {
    if (mode == "ransac") {
        return FittingStrategy::RANSAC;
    } else if (mode == "lmeds") {
        return FittingStrategy::LMEDS;
    }
    return FittingStrategy::STANDARD;
}

std::string getOutlierRejectionState_string(OutlierRejectionState state) {
    if (state == OutlierRejectionState::SIMPLE) {
        return "simple";
    } else if (state == OutlierRejectionState::HOMOGRAPHY_FIT) {
        return "homography";
    }
    return "undefined";
}

std::string getFittingStrategy_string(FittingStrategy mode) {
    if (mode == FittingStrategy::RANSAC) {
        return "ransac";
    } else if (mode == FittingStrategy::LMEDS) {
        return "lmeds";
    }
    return "standard";
}

std::string getFeatureExtractionMethod_string(FeatureExtractionMethod method) {
    if (method == FeatureExtractionMethod::FAST) {
        return "FAST";
    } else if (method == FeatureExtractionMethod::SHI_TOMASI) {
        return "shi-tomasi";
    }
    return "undefined";
}

std::string getCornerSubPixState_string(CornerSubPixState state) {
    if (state == CornerSubPixState::ENABLED) {
       return "enabled";
    } else if (state == CornerSubPixState::DISABLED) {
        return "disabled";
    }
    return "undefined";
}

std::string getOptimizationWeightingStrategy_string(OptimizationWeightStrategy mode) {
    if (mode == OptimizationWeightStrategy::ADAPTIVE_WEIGHTING) {
        return "adaptive";
    } else if (mode == OptimizationWeightStrategy::CONSTANT_WEIGHTING) {
        return "constant";
    }
    return "undefined";
}

std::string getOptimizationMode_string(OptimizationMode mode) {
    if (mode == OptimizationMode::OFFLINE) {
        return "offline";
    } else if (mode == OptimizationMode::ONLINE) {
        return "online";
    }
    return "undefined";
}

}   // namespace config

}   // namespace meshflow
