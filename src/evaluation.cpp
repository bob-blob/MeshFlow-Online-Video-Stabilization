#include "evaluation.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <vector>

namespace meshflow {

using cv::VideoCapture;
using cv::Mat;
using std::vector;

Evaluation::Evaluation() {}

EVALUATION_SCORE Evaluation::distortion(const std::string& originalVideoFilename,
                                        const std::string& stabilizedVideoFilename) const {
    VideoCapture capOriginal(originalVideoFilename),
                 capStabilized(stabilizedVideoFilename);
    int frameCount = static_cast<int>(capOriginal.get(cv::CAP_PROP_FRAME_COUNT));

    Mat frameOriginal, frameStabilized;

    capOriginal >> frameOriginal;
    capStabilized >> frameStabilized;


    for (int i = 0; i < frameCount; ++i) {
        // Or better is to find homography between extracted keyPoints

        vector<float> corners, flowCorners;
        cv::goodFeaturesToTrack(frameOriginal, corners, 500, 0.01, 7);

        //cv::calcOptic
        Mat H = cv::findHomography(frameOriginal, frameStabilized);

    }


}

}
