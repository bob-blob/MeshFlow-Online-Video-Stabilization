#include "OpticalFlow.h"

namespace meshflow {

void FastKltTracker::getMatches(const Mat &source, const Mat &target,
                                      vector<Point2f> &sourceFeatures, vector<Point2f> &targetFeatures) {
    Mat sourceGray = Mat::zeros(source.size(), CV_8UC1);
    Mat targetGray = Mat::zeros(target.size(), CV_8UC1);
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY); // They have RGB2GRAY
    cv::cvtColor(target, targetGray, cv::COLOR_BGR2GRAY);

    vector<Point2f> sourceFeatureSet;
    int maxNum = 10000;
    cv::goodFeaturesToTrack(sourceGray, sourceFeatureSet, maxNum, 0.05, 5);
    cv::cornerSubPix(sourceGray, sourceFeatureSet, cv::Size(15, 15),
                     cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

    vector<Point2f> targetFeatureSet;
    vector<uchar> status;
    vector<float> targetErrorSet;
    cv::calcOpticalFlowPyrLK(sourceGray, targetGray, sourceFeatureSet, targetFeatureSet, status, targetErrorSet, cv::Size(15, 15));

    /// Check Boundaries of extracted corners
    for (size_t p = 0; p < targetErrorSet.size(); ++p) {
        if (targetErrorSet[p] > 100   || targetFeatureSet[p].x < 0 || targetFeatureSet[p].y < 0 ||
            targetFeatureSet[p].x > sourceGray.cols || targetFeatureSet[p].y > sourceGray.rows)
        {
            status[p] = 0;
        }
    }

    /// Remove obvious outliers
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1)
        {
            sourceFeatures.push_back(sourceFeatureSet[i]);
            targetFeatures.push_back(targetFeatureSet[i]);
        }
    }
}

}
