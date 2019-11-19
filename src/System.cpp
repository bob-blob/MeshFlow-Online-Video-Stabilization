#include "System.h"

#include "OpticalFlow.h"

namespace meshflow {

using std::string;
using namespace cv;

System::System(string videoFilename, string configFilename)
    :   videoFilename(videoFilename)
{
    videoIn = VideoCapture(videoFilename);

    if (!configFilename.empty()) {
        // config::saveConfigs(configFilename);
        mainConfig = config::loadConfigs(configFilename);
    }
    int width  = static_cast<int>(videoIn.get(cv::CAP_PROP_FRAME_WIDTH)),
        height = static_cast<int>(videoIn.get(cv::CAP_PROP_FRAME_HEIGHT));
    frameResolution = Size(width, height);

    cellSize.width = (frameResolution.width / mainConfig.meshFlowConfig.gridSize.width);
    cellSize.height = (frameResolution.height / mainConfig.meshFlowConfig.gridSize.height);

    HORIZONTAL_BORDER = 30;
    VERTICAL_BORDER = (HORIZONTAL_BORDER * frameResolution.width) / frameResolution.height;
}

void System::stabilize()
{
    videoIn.set(cv::CAP_PROP_POS_FRAMES, 0);
    Mat prevFrame, prevFrameGray;
    videoIn >> prevFrame;
    cv::cvtColor(prevFrame, prevFrameGray, cv::COLOR_BGR2GRAY);
    int frameCount = static_cast<int>(videoIn.get(cv::CAP_PROP_FRAME_COUNT));

    // motion meshes in x-direction and y-direction
    xPaths.push_back(Mat::zeros(prevFrame.rows/cellSize.height + 1, prevFrame.cols/cellSize.width + 1, CV_64F));
    yPaths.push_back(Mat::zeros(prevFrame.rows/cellSize.height + 1, prevFrame.cols/cellSize.width + 1, CV_64F));

    int currFrameIdx{1};

    lambdas.push_back(std::make_pair(100, 100));

    MeshFlow meshFlow(mainConfig.meshFlowConfig);

    while (currFrameIdx < frameCount) {
        Mat nextFrame, nextFrameGray;
        videoIn >> nextFrame;
        if (nextFrame.empty()) {
            break;
        }
        cv::cvtColor(nextFrame, nextFrameGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2d> prevFrameCorners, nextFrameCorners;

        std::vector<uchar> status;
        OpticalFlow::trackKLT(
                    prevFrameGray, nextFrameGray, prevFrameCorners, nextFrameCorners, mainConfig.optFlowConfig);

        //for (size_t i = 0; i < prevFrameCorners.size(); ++i) {
        //   cv::circle(prevFrame, prevFrameCorners[i], 1, cv::Scalar(124, 252, 0), -1);
        //}

        // Estimate motion mesh for old_frame
        Mat xMotionMesh = Mat::zeros(frameResolution.height/cellSize.height + 1, frameResolution.width/cellSize.width + 1, CV_64F),
            yMotionMesh = Mat::zeros(frameResolution.height/cellSize.height + 1, frameResolution.width/cellSize.width + 1, CV_64F);

        meshFlow.motionPropagate(
                    prevFrameCorners, nextFrameCorners, nextFrame, xMotionMesh, yMotionMesh, lambdas, cv::Size(cellSize.width, cellSize.height));
        xMotionMeshes.push_back(xMotionMesh);
        yMotionMeshes.push_back(yMotionMesh);

        meshFlow.generateVertexProfiles(xMotionMesh, yMotionMesh, xPaths, yPaths);

        //imshow("disp", oldFrame);
        //char c = static_cast<char>(cv::waitKey(33));
        //if (c == 27) {
        //   break;
        //}

        Log("frame processed: " << currFrameIdx);
        currFrameIdx++;
        prevFrame = nextFrame.clone();
        prevFrameGray = nextFrameGray.clone();
    }

    Log("1. Video is processed.");
    bool optimizerMode = (mainConfig.optimConfig.mode == config::OptimizationMode::ONLINE) ? true : false;
    optimize(optimizerMode);

    // Get updated mesh warps
    this->getFrameWarp();

    // Apply updated mesh warps & save the results
    this->generateStabilizedVideo(videoIn);
}


void System::optimize(bool online) {

    if (online) {
        Optimizer::onlinePathOptimization(
                    xPaths, smoothXPaths, mainConfig.optimConfig.frameBufferSize, mainConfig.optimConfig.iterations,
                    mainConfig.optimConfig.windowSize, mainConfig.optimConfig.beta);
        Optimizer::onlinePathOptimization(
                    yPaths, smoothYPaths, mainConfig.optimConfig.frameBufferSize, mainConfig.optimConfig.iterations,
                    mainConfig.optimConfig.windowSize, mainConfig.optimConfig.beta);
    } else {
        Optimizer::offlinePathOptimization(
                    xPaths, smoothXPaths, lambdas, mainConfig.optimConfig.iterations, mainConfig.optimConfig.windowSize);
        Optimizer::offlinePathOptimization(
                    yPaths, smoothYPaths, lambdas, mainConfig.optimConfig.iterations, mainConfig.optimConfig.windowSize);
    }

    Log("2. Vertex profiles are stabilized (smoothed).");
}

void System::getFrameWarp()
{
    std::cout << "xMotionMeshes size: " << xMotionMeshes.size() << std::endl;
    std::cout << "xPaths size: " << xPaths.size() << std::endl;

    xMotionMeshes.push_back(xMotionMeshes.back());
    yMotionMeshes.push_back(yMotionMeshes.back());

    for (size_t i = 0; i < xPaths.size(); ++i) {
        newXMotionMeshes.push_back(smoothXPaths[i] - xPaths[i]);
        newYMotionMeshes.push_back(smoothYPaths[i] - yPaths[i]);
    }

    Log("3. New frame warp obtained.");
}

void System::generateStabilizedVideo(VideoCapture &cap)
{
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    int frameRate = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));


    // get video parameters
    cv::VideoWriter out = cv::VideoWriter(videoFilename+"_stable.avi", CV_FOURCC('M', 'J', 'P', 'G'), frameRate, cv::Size(frameWidth*2, frameHeight));

    int frameNum{0};
    MeshFlow meshFlow;
    while (frameNum < frameCount-1) { // Problem: was just frameCount and worked with all videos except video-calib (video-calib worked)
        Mat frame, frameGray;
        cap >> frame;

        // Reconstruct from frames
        Mat xMotionMesh = xMotionMeshes[frameNum];
        Mat yMotionMesh = yMotionMeshes[frameNum];
        Mat newXMotionMesh = newXMotionMeshes[frameNum];
        Mat newYMotionMesh = newYMotionMeshes[frameNum];

        // Mesh warping
        Mat newFrame;
        meshFlow.meshWarpFrame(frame, newXMotionMesh, newYMotionMesh, newFrame, cv::Size(cellSize.width, cellSize.height));
        // new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]

        /// resize image - apply crop (uncomment)
        //newFrame = newFrame(cv::Range(HORIZONTAL_BORDER, frame.rows-HORIZONTAL_BORDER), cv::Range(VERTICAL_BORDER, frame.cols-VERTICAL_BORDER));
        //cv::resize(newFrame, newFrame, cv::Size(frame.cols, frame.rows), cv::INTER_CUBIC);

        /// Concatenate both images
        Mat outputFrame(frameHeight, 2*frameWidth, frame.type());
        cv::hconcat(frame, newFrame, outputFrame);

        out.write(outputFrame);

        // Draw old motion vectors and draw new motion vectors
        // TODO: to do or not to do
        int radius = 5;
        for (int i = 0; i < xMotionMesh.cols; ++i) {
            for (int j = 0; j < xMotionMesh.rows; ++j) {
                double theta = atan2(yMotionMesh.at<double>(i, j), xMotionMesh.at<double>(i, j));
                cv::line(frame, cv::Point2d(j*cellSize.width, i*cellSize.height),
                                cv::Point2d((j*cellSize.width + radius * cos(theta)),
                                            (i*cellSize.height + radius * sin(theta))), 2);
            }
        }
        std::string path = "../results/old_motion_vectors/",
                    extension = ".jpg";
        cv::imwrite(path + std::to_string(frameNum) + extension, frame);

        for (int i = 0; i < newXMotionMesh.cols; ++i) {
            for (int j = 0; j < newXMotionMesh.rows; ++j) {
                double theta = atan2(newYMotionMesh.at<double>(i, j), newXMotionMesh.at<double>(i, j));
                cv::line(newFrame, cv::Point2d(j*cellSize.width, i*cellSize.height),
                                cv::Point2d((j*cellSize.width + radius * cos(theta)),
                                            (i*cellSize.height + radius * sin(theta))), 2);
            }
        }
        cv::imwrite("../results/new_motion_vectors/" + std::to_string(frameNum) + extension, newFrame);

        frameNum++;
    }
    Log("4. Finished writing warped video.");
    cap.release();
    out.release();

}



}
