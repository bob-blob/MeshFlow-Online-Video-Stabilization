#include "Stabilizer.h"

namespace meshflow {

void Stabilizer::readVideo(VideoCapture& cap) {

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    // Take first frame
    Mat oldFrame, oldGray;
    cap >> oldFrame;
    cv::cvtColor(oldFrame, oldGray, cv::COLOR_BGR2GRAY);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cellWidth = width / 4;
    cellHeight = width / 4;

    // Preserve aspect ratio
    HORIZONTAL_BORDER = 30;
    VERTICAL_BORDER = (HORIZONTAL_BORDER * oldGray.cols) / oldGray.rows;

    // motion meshes in x-direction and y-direction
    xPaths.push_back(Mat::zeros(oldFrame.rows/PIXELS_Y + 1, oldFrame.cols/PIXELS_X + 1, CV_64F));
    yPaths.push_back(Mat::zeros(oldFrame.rows/PIXELS_Y + 1, oldFrame.cols/PIXELS_X + 1, CV_64F));

    int frameNum{1};

    lambdas.push_back(std::make_pair(1, 1));

    while (frameNum < frameCount) {
        Mat frame, frameGray;
        cap >> frame;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> p0, p1;
        extractFeaturesFromGrid(oldGray, p0); // rich feature extraction
        //cv::goodFeaturesToTrack(oldGray, p0, 1000, 0.3, 7, Mat(), 7);

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, status, err, cv::Size(15, 15), 2, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));

        /// select good points: outlier rejection
        std::vector<cv::Point2d> goodOld, goodNew;
        outlierRejection(status, p0, p1, goodOld, goodNew);
        //simpleRejection(status, p0, p1, goodOld, goodNew);

        // Estimate motion mesh for old_frame
        Mat xMotionMesh = Mat::zeros(oldFrame.rows/PIXELS_Y + 1, oldFrame.cols/PIXELS_X + 1, CV_64F),
            yMotionMesh = Mat::zeros(oldFrame.rows/PIXELS_Y + 1, oldFrame.cols/PIXELS_X + 1, CV_64F);

        MeshFlow::motionPropagate(goodOld, goodNew, frame, xMotionMesh, yMotionMesh, lambdas, cv::Size(PIXELS_X, PIXELS_Y));
        xMotionMeshes.push_back(xMotionMesh);
        yMotionMeshes.push_back(yMotionMesh);
        //Log(xMotionMesh);
        // Generate vertex profiles
        MeshFlow::generateVertexProfiles(xMotionMesh, yMotionMesh, xPaths, yPaths);

        frameNum++;
        oldFrame = frame.clone();
        oldGray = frameGray.clone();
    }

    Log("1. Video is processed.");
}

void Stabilizer::stabilize(bool online) {

    if (online) {
        Optimizer::onlinePathOptimization(xPaths, smoothXPaths, 30);
        Log("smoothX!!!");
        Optimizer::onlinePathOptimization(yPaths, smoothYPaths, 30);
        Log("smoothY!!!");
    } else {
        Optimizer::offlinePathOptimization(xPaths, smoothXPaths, lambdas);
        Log("smoothX!!!");
        Optimizer::offlinePathOptimization(yPaths, smoothYPaths, lambdas);
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

    for (size_t i = 0; i < xPaths.size(); ++i) {
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
    cv::VideoWriter out = cv::VideoWriter(filename+"_stable.avi", CV_FOURCC('M', 'J', 'P', 'G'), frameRate, cv::Size(frameWidth*2, frameHeight));

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
        MeshFlow::meshWarpFrame(frame, newXMotionMesh, newYMotionMesh, newFrame, cv::Size(PIXELS_X, PIXELS_Y));
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

        frameNum++;
    }
    Log("4. Finished writing warped video.");
    cap.release();
    out.release();

}

void Stabilizer::extractFeaturesFromGrid(const Mat &frame, std::vector<cv::Point2f>& corners) {
    vector<Mat> grid;
    int i{0};
    vector<cv::Point2f> cellCorners[16], newFrameCorners;

    // The part which extract features for each grid cell
    for (int y = 0; y <= height - cellHeight; y += cellHeight)
    {
        for (int x = 0; x <= width - cellWidth; x += cellWidth)
        {

            cv::Rect region;
            region.x = x; region.y = y;
            region.width = static_cast<int>(cellWidth);
            region.height = static_cast<int>(cellHeight);

            Mat cell = frame(region);
            grid.push_back(cell);

            goodFeaturesToTrack(cell, cellCorners[i], 200, 0.001, 7, Mat(), 3);

            // Fix points to account cell coordinates w.r.t. image coordinates
            for(auto & j : cellCorners[i])
            {
                j = cv::Point2f(j.x + x, j.y + y);
            }


            // Now get all points in one array
            corners.insert(corners.end(), cellCorners[i].begin(), cellCorners[i].end());
            i++;
        }
    }
}

void Stabilizer::simpleRejection(const vector<uchar> &status, const vector<Point2f> &frameCorners,
                                 const vector<Point2f> &newFrameCorners,
                                 vector<Point2d> &old_points, vector<Point2d> &new_points) {

    for (int i = 0; i < status.size(); ++i) {
        if (status[i] == 1) {
            new_points.push_back(newFrameCorners[i]);
            old_points.push_back(frameCorners[i]);
        }
    }
}


void Stabilizer::outlierRejection(const vector<uchar> &status, const vector<Point2f> &frameCorners,
                                  const vector<Point2f> &newFrameCorners,
                                  vector<Point2d> &old_points, vector<Point2d> &new_points) {
    // Brush off invalid points after optical flow
    vector<Point2d> goodNew_temp, goodOld_temp;
    for (int i = 0; i < status.size(); ++i) {
        if (status[i] == 1) {
            goodNew_temp.push_back(newFrameCorners[i]);
            goodOld_temp.push_back(frameCorners[i]);
        }
    }

    int i{0};
    // TODO: Parallelize code
    for (int y = 0; y <= height - cellHeight; y += cellHeight) {
        for (int x = 0; x <= width - cellWidth; x += cellWidth) {

            cv::Rect region;
            region.x = x; region.y = y;
            region.width = static_cast<int>(cellWidth);
            region.height = static_cast<int>(cellHeight);

            vector<Point2d> binOldPoints, binNewPoints;
            for (size_t i = 0; i < goodOld_temp.size(); ++i)
            {
                if(region.contains(goodOld_temp[i]) && region.contains(goodNew_temp[i])) {
                    binOldPoints.push_back(goodOld_temp[i]);
                    binNewPoints.push_back(goodNew_temp[i]);
                }
            }

            vector<uchar> status_new;
            Mat H = findHomography(binOldPoints, binNewPoints, status_new, cv::RANSAC, 2);

            vector<Point2f> cleanOldPoints, cleanNewPoints;
            for (size_t i = 0; i < binOldPoints.size(); ++i)
            {
                if (status_new[i]) {
                    cleanOldPoints.push_back(binOldPoints[i]);
                    cleanNewPoints.push_back(binNewPoints[i]);
                }
            }

            // Now get all points in one array
            old_points.insert(old_points.end(), cleanOldPoints.begin(), cleanOldPoints.end());
            new_points.insert(new_points.end(), cleanNewPoints.begin(), cleanNewPoints.end());
            i++;
        }
    }
}

}
