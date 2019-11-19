#include "OpticalFlow.h"

namespace meshflow {

using cv::Mat;
using cv::Point2f;
using cv::Point2d;
using std::vector;

using namespace config;

namespace OpticalFlow {

void extractCornersShiTomasi(
    const cv::Mat& frame,
    vector<Point2f>& extractedCorners,
    const GoodFeaturesToTrackParameters& cornerExtractionParams,
    CornerSubPixState subPixState,
    const CornerSubPixParameters& cornerSubpixParams)
{
    cv::goodFeaturesToTrack(
                frame,
                extractedCorners,
                cornerExtractionParams.maxCorners,
                cornerExtractionParams.qualityLevel,
                cornerExtractionParams.minDistance,
                cv::noArray(),
                cornerExtractionParams.blockSize,
                cornerExtractionParams.useHarris,
                cornerExtractionParams.k);
    if (subPixState == CornerSubPixState::ENABLED)
        cv::cornerSubPix(
                    frame,
                    extractedCorners,
                    cornerSubpixParams.winSize,
                    cornerSubpixParams.zeroZone,
                    cornerSubpixParams.termCriteria);
}

void trackKLT(const Mat& sourceGray,
              const Mat& targetGray,
              vector<Point2d>& sourceFeatures,
              vector<Point2d>& targetFeatures,
              const OpticalFlowConfiguration& optFlowConfig)
{
    vector<Point2f> sourceFeatureSet;

    if (optFlowConfig.extractionMethod == FeatureExtractionMethod::SHI_TOMASI) {
        extractFeaturesFromGrid(
                    sourceGray, sourceFeatureSet, optFlowConfig.kltParams.trackingGridSize, optFlowConfig.goodFeaturesParams,
                    optFlowConfig.subpixState, optFlowConfig.subpixParams);
    } else if (optFlowConfig.extractionMethod == FeatureExtractionMethod::FAST) {
        extractFASTFeaturesFromGrid(
                    sourceGray, sourceFeatureSet, optFlowConfig.kltParams.trackingGridSize, optFlowConfig.fastParams);
    }

    vector<Point2f> targetFeatureSet;
    vector<uchar> status;
    vector<float> targetErrorSet;
    cv::calcOpticalFlowPyrLK(
                sourceGray,
                targetGray,
                sourceFeatureSet,
                targetFeatureSet,
                status,
                targetErrorSet,
                optFlowConfig.kltParams.winSize,
                optFlowConfig.kltParams.maxPyramidLevel,
                optFlowConfig.kltParams.termCriteria);

    FramePairCorners currFeatureSet = makeFramePairCorners(sourceFeatureSet, targetFeatureSet),
                     filteredFeatureSet;

    if (optFlowConfig.outlierParams.state == OutlierRejectionState::HOMOGRAPHY_FIT) {
        outlierRejection(sourceGray, currFeatureSet, status, filteredFeatureSet, optFlowConfig.outlierParams);
    } else if (optFlowConfig.outlierParams.state == OutlierRejectionState::SIMPLE) {
        rejectBadCorners(currFeatureSet, status, filteredFeatureSet);
    }

    sourceFeatures = vector<Point2d>{filteredFeatureSet.first.begin(), filteredFeatureSet.first.end()};
    targetFeatures = vector<Point2d>{filteredFeatureSet.second.begin(), filteredFeatureSet.second.end()};
}

void extractFASTFeaturesFromGrid(const Mat &frame,
                                 std::vector<cv::Point2f>& corners,
                                 cv::Size gridSize,
                                 const FASTParameters& fastParams) {

    size_t cellCount{0};
    int cellHeight = frame.rows / gridSize.height,
        cellWidth = frame.cols / gridSize.width;

    vector<vector<Point2f>> cellCorners(16);
    vector<vector<cv::KeyPoint>> keyPoints(16);

    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(fastParams.threshold,
                                                                                fastParams.nonmaxSuppression,
                                                                                fastParams.type);

    // The part which extract features for each grid cell
    for (int y = 0; y <= frame.rows - cellHeight; y += cellHeight) {
        for (int x = 0; x <= frame.cols - cellWidth; x += cellWidth) {
            cv::Rect region{ x, y, static_cast<int>(cellWidth), static_cast<int>(cellHeight)};
            Mat cell = frame(region);

            detector->detect(cell, keyPoints[cellCount]);

            if (keyPoints[cellCount].size() < 20) // if the region contains a little key points (sky, road)
            {
                cv::Ptr<cv::FastFeatureDetector> newDetector = cv::FastFeatureDetector::create(3);
                keyPoints[cellCount].clear();
                newDetector->detect(cell, keyPoints[cellCount]);
            }

            for (size_t j = 0; j < keyPoints[cellCount].size(); ++j) {
                cellCorners[cellCount].push_back(keyPoints[cellCount][j].pt);
            }

            // Fix points to account cell coordinates w.r.t. image coordinates
            for(auto& j : cellCorners[cellCount])
                j = cv::Point2f(j.x + x, j.y + y);

            // Now get all points in one array
            corners.insert(corners.end(), cellCorners[cellCount].begin(), cellCorners[cellCount].end());
            cellCount++;
        }
    }
}

void extractFeaturesFromGrid(const Mat &frame,
                             std::vector<cv::Point2f>& corners,
                             cv::Size gridSize,
                             const GoodFeaturesToTrackParameters& cornerExtractionParams,
                             config::CornerSubPixState subPixState,
                             const CornerSubPixParameters& cornerSubpixParams)
{
    size_t cellCount{0};
    size_t numOfCells = static_cast<size_t>(gridSize.width*gridSize.height);
    vector<vector<cv::Point2f>> cellCorners(numOfCells);

    int cellWidth = frame.cols / gridSize.width,
        cellHeight = frame.rows / gridSize.height;

    // The part which extract features for each grid cell
    for (int y = 0; y <= frame.rows - cellHeight; y += cellHeight) {
        for (int x = 0; x <= frame.cols - cellWidth; x += cellWidth) {
            cv::Rect region{x, y, static_cast<int>(cellWidth), static_cast<int>(cellHeight)};
            Mat cell = frame(region);

            extractCornersShiTomasi(frame, cellCorners[cellCount], cornerExtractionParams, subPixState, cornerSubpixParams);

            // Fix points to account cell coordinates w.r.t. image coordinates
            for(auto &j : cellCorners[cellCount])
                j = cv::Point2f(j.x + x, j.y + y);

            // Now get all points in one array
            corners.insert(corners.end(), cellCorners[cellCount].begin(), cellCorners[cellCount].end());
            cellCount++;
        }
    }
}

void outlierRejection(const Mat& frame,
                      const FramePairCorners& corners,
                      const vector<uchar>& status,
                      FramePairCorners& filteredCorners,
                      const OutlierRejectionParameters& outlierParams) {

    FramePairCorners intermediateResultCorners;
    rejectBadCorners(corners, status, intermediateResultCorners);

    int cellHeight = frame.rows / outlierParams.gridSize.height,
        cellWidth = frame.cols / outlierParams.gridSize.width;

    for (int y = 0; y <=  frame.rows - cellHeight; y += cellHeight) {
        for (int x = 0; x <= frame.cols - cellWidth; x += cellWidth) {
            cv::Rect region{ x, y, static_cast<int>(cellWidth), static_cast<int>(cellHeight)};

            FramePairCorners binPoints;
            for (size_t i = 0; i < intermediateResultCorners.first.size(); ++i)
            {
                if(region.contains(intermediateResultCorners.first[i]) &&
                   region.contains(intermediateResultCorners.second[i])) {
                    binPoints.first.push_back(intermediateResultCorners.first[i]);
                    binPoints.second.push_back(intermediateResultCorners.second[i]);
                }
            }

            if(binPoints.first.size() <= 3 || binPoints.second.size() <=3)
            {
                filteredCorners.first.insert(filteredCorners.first.end(),
                                             binPoints.first.begin(),
                                             binPoints.first.end());
                filteredCorners.second.insert(filteredCorners.second.end(),
                                              binPoints.second.begin(),
                                              binPoints.second.end());
                continue;
            }

            vector<uchar> status_new;
            int method;
            if (outlierParams.homographyFit.fitStrategy == FittingStrategy::RANSAC)
                method = cv::RANSAC;
            else if (outlierParams.homographyFit.fitStrategy == FittingStrategy::LMEDS)
                method = cv::LMEDS;
            else
                method = 0;

            Mat H = findHomography(binPoints.first, binPoints.second, status_new, method, outlierParams.homographyFit.rejectionThreshold);

            FramePairCorners cleanPoints;
            rejectBadCorners(binPoints, status_new, cleanPoints);

            // Now get all points in one array
            filteredCorners.first.insert(filteredCorners.first.end(),
                                         cleanPoints.first.begin(),
                                         cleanPoints.first.end());
            filteredCorners.second.insert(filteredCorners.second.end(),
                                          cleanPoints.second.begin(),
                                          cleanPoints.second.end());
        }
    }
}

void rejectBadCorners(const FramePairCorners& corners,
                      const vector<uchar>& status,
                      FramePairCorners& filteredCorners) {
    for (size_t i = 0; i < corners.first.size(); ++i) {
        if (status[i] == 1) {
            filteredCorners.first.push_back(corners.first[i]);
            filteredCorners.second.push_back(corners.second[i]);
        }
    }
}

}

}
