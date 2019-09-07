#include "Stabilizer.h"

namespace meshflow {

void Stabilizer::readVideo(VideoCapture& cap) {
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    // Take first frame
    Mat oldFrame, oldGray;
    cap >> oldFrame;
    cv::cvtColor(oldFrame, oldGray, cv::COLOR_BGR2GRAY);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Preserve aspect ratio
    HORIZONTAL_BORDER = 30;
    VERTICAL_BORDER = (HORIZONTAL_BORDER * oldGray.cols) / oldGray.rows;

    // motion meshes in x-direction and y-direction

}

void Stabilizer::stabilize(const Mat& xPaths, const Mat& yPaths,
               Mat& smoothXPaths, Mat& smoothYPaths) {

}

}
