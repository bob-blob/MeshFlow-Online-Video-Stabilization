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
    xPaths.push_back(Mat::zeros(oldFrame.rows/PIXELS, oldFrame.cols/PIXELS, CV_64F));
    yPaths.push_back(Mat::zeros(oldFrame.rows/PIXELS, oldFrame.cols/PIXELS, CV_64F));

    int frameNum{1};

    while (frameNum < frameCount) {
        Mat frame, frameGray;
        cap >> frame;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> p0, p1;
        cv::goodFeaturesToTrack(oldGray, p0, 1000, 0.3, 7, Mat(), 7);

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, status, err, cv::Size(15, 15), 2, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

        // select good points
        std::vector<cv::Point2d> goodOld, goodNew;

        for (int i = 0; i < status.size(); ++i) {
            if (status[i] == 1) {
                goodNew.push_back(p1[i]);
                goodOld.push_back(p0[i]);
            }
        }

        // Estimate motion mesh for old_frame
        Mat xMotionMesh, yMotionMesh;
        MeshFlow::motionPropagate(goodOld, goodNew, frame, xMotionMesh, yMotionMesh);
        xMotionMeshes.push_back(xMotionMesh);
        yMotionMeshes.push_back(yMotionMesh);

        // Generate vertex profiles
        MeshFlow::generateVertexProfiles(xMotionMesh, yMotionMesh, xPaths, yPaths);

        frameNum++;
        oldFrame = frame.clone();
        oldGray = frameGray.clone();
    }

    Log("1. Video is processed.");
}

void Stabilizer::stabilize(bool online) {
    // std::cout << xPaths[2] << std::endl;
    if (online) {
        Optimizer::onlinePathOptimization(xPaths, smoothXPaths, 30);
        Log("smoothX!!!");
        Optimizer::onlinePathOptimization(yPaths, smoothYPaths, 30);
        Log("smoothY!!!");
    } else {
        Optimizer::offlinePathOptimization(xPaths, smoothXPaths);
        Log("smoothX!!!");
        Optimizer::offlinePathOptimization(yPaths, smoothYPaths);
        Log("smoothY!!!");
    }

    Log("2. Vertex profiles are stabilized.");
}

void Stabilizer::getFrameWarp()
{
    std::cout << "xMotionMeshes size: " << xMotionMeshes.size() << std::endl;
    std::cout << "xPaths size: " << xPaths.size() << std::endl;

    xMotionMeshes.push_back(xMotionMeshes.back());
    yMotionMeshes.push_back(yMotionMeshes.back());

    for (int i = 0; i < xPaths.size(); ++i) {
        newXMotionMeshes.push_back(smoothXPaths[i] - xPaths[i]);
        newYMotionMeshes.push_back(smoothYPaths[i] - yPaths[i]);
    }

    Log("3. New frame warp obtained.");
}

void Stabilizer::generateStabilizedVideo(VideoCapture &cap)
{
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    int frameRate = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // get video parameters
    cv::VideoWriter out = cv::VideoWriter("../stable.avi", CV_FOURCC('X', 'V', 'I', 'D'), frameRate, cv::Size(frameWidth*2, frameHeight));

    int frameNum{0};
    while (frameNum < frameCount) {
        Mat frame, frameGray;
        cap >> frame;

        // Reconstruct from frames
        Mat xMotionMesh = xMotionMeshes[frameNum];
        Mat yMotionMesh = yMotionMeshes[frameNum];
        Mat newXMotionMesh = newXMotionMeshes[frameNum];
        Mat newYMotionMesh = newYMotionMeshes[frameNum];

        // Mesh warping
        Mat newFrame;
        MeshFlow::meshWarpFrame(frame, newXMotionMesh, newYMotionMesh, newFrame);
        // new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]

        /// resize image - apply crop (uncomment)
        newFrame = newFrame(cv::Range(HORIZONTAL_BORDER, frame.rows-HORIZONTAL_BORDER), cv::Range(VERTICAL_BORDER, frame.cols-VERTICAL_BORDER));
        cv::resize(newFrame, newFrame, cv::Size(frame.cols, frame.rows), cv::INTER_CUBIC);

        // Write images
        Mat outputFrame(frameHeight, 2*frameWidth, frame.type());
        cv::hconcat(frame, newFrame, outputFrame);

        out.write(outputFrame);

        // Draw old motion vectors and draw new motion vectors
        // TODO: to do or not to do

        frameNum++;
    }
    Log("4. Finished writing warped video.");
    cap.release();
    out.release();

}

}
