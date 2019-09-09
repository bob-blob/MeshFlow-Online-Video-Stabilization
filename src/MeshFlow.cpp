#include "MeshFlow.h"

namespace meshflow {

void MeshFlow::transformPoint(const Mat& H, const Point2d& point, Point2d& result) {

    double a = H.at<double>(0, 0) * point.x + H.at<double>(0, 1) * point.y + H.at<double>(0, 2);
    double b = H.at<double>(1, 0) * point.x + H.at<double>(1, 1) * point.y + H.at<double>(1, 2);
    double c = H.at<double>(2, 0) * point.x + H.at<double>(2, 1) * point.y + H.at<double>(2, 2);

    result = Point2d(a/c, b/c);
}

void MeshFlow::motionPropagate(const vector<Point2d> oldPoints,
                               const vector<Point2d> newPoints,
                               const Mat& oldFrame,
                               Mat& xMotionMesh, Mat& yMotionMesh) {

    int cols = static_cast<int>(oldFrame.cols / PIXELS);
    int rows = static_cast<int>(oldFrame.rows / PIXELS);

    Mat H = cv::findHomography(oldPoints, newPoints, cv::RANSAC);

    Map xMotion, yMotion;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Point2d point(PIXELS*j, PIXELS*i);
            Point2d pointTransform;
            transformPoint(H, point, pointTransform);

            xMotion[std::make_pair(i, j)].push_back(point.x - pointTransform.x);
            yMotion[std::make_pair(i, j)].push_back(point.y - pointTransform.y);
        }
    }

    // Distribute feature motion vectors
    Map temp_xMotion, temp_yMotion;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Point2d vertex(PIXELS*j, PIXELS*i);
            for (int k = 0; k < oldPoints.size(); ++k) {
                double distance = sqrt(pow(vertex.x - oldPoints[k].x, 2) + pow(vertex.y - oldPoints[k].y, 2));
                if (distance < RADIUS) {
                    Point2d pointTransform;
                    transformPoint(H, oldPoints[k], pointTransform);
                    temp_xMotion[std::make_pair(i, j)].push_back(newPoints[k].x - pointTransform.x);
                    temp_yMotion[std::make_pair(i, j)].push_back(newPoints[k].y - pointTransform.y);
                }
            }
        }
    }

    // Apply first median filter on obtained motion for each vertex
    xMotionMesh.create(rows, cols, CV_64F);
    yMotionMesh.create(rows, cols, CV_64F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::pair<int, int> p = std::make_pair(i, j);

            Map::iterator itX = temp_xMotion.find(p);
            if (itX != temp_xMotion.end())
            {
                std::sort(itX->second.begin(), itX->second.end());
                xMotionMesh.at<double>(i, j) = xMotion[p].back() + temp_xMotion[p][temp_xMotion[p].size()/2];
            } else {
                xMotionMesh.at<double>(i, j) = xMotion[p].back();
            }

            Map::iterator itY = temp_yMotion.find(p);
            if (itY != temp_yMotion.end())
            {
                std::sort(itY->second.begin(), itY->second.end());
                yMotionMesh.at<double>(i, j) = yMotion[p].back() + temp_yMotion[p][temp_yMotion[p].size()/2];
            } else {
                yMotionMesh.at<double>(i, j) = yMotion[p].back();
            }
        }
    }
    // TODO: Apply the second medial filter over the motion field to discard the outliers
    // median kernel [3x3]
    ///!!!!!!

    xMotionMesh.convertTo(xMotionMesh, CV_32F);

    //std::cout << xMotionMesh << std::endl;
    cv::medianBlur(xMotionMesh, xMotionMesh, 3);
    //::cout << xMotionMesh << std::endl;
    xMotionMesh.convertTo(xMotionMesh, CV_64F);

    yMotionMesh.convertTo(yMotionMesh, CV_32F);
    cv::medianBlur(yMotionMesh, yMotionMesh, 3);
    yMotionMesh.convertTo(yMotionMesh, CV_64F);

}

void MeshFlow::generateVertexProfiles(const Mat& xMotionMesh,
                                      const Mat& yMotionMesh,
                                      vector<Mat>& xPaths, vector<Mat>& yPaths) {
    xPaths.push_back(xMotionMesh + xPaths.back());
    yPaths.push_back(yMotionMesh + yPaths.back());
}

void MeshFlow::meshWarpFrame(const Mat& frame,
                             const Mat& xMotionMesh,
                             const Mat& yMotionMesh,
                             Mat& warpedFrame) {

    Mat mapX = Mat::zeros(frame.rows, frame.cols, CV_32F);
    Mat mapY = Mat::zeros(frame.rows, frame.cols, CV_32F);

    for (int i = 0; i < xMotionMesh.rows-1; ++i) {
        for (int j = 0; j < xMotionMesh.cols-1; ++j) {

            vector<Point2d> src = { Point2d(j*PIXELS, i*PIXELS),
                                    Point2d(j*PIXELS, (i+1)*PIXELS),
                                    Point2d((j+1)*PIXELS, i*PIXELS),
                                    Point2d((j+1)*PIXELS, (i+1)*PIXELS)};

            vector<Point2d> dst = { Point2d(j*PIXELS+xMotionMesh.at<double>(i, j),
                                            i*PIXELS+yMotionMesh.at<double>(i, j)),
                                    Point2d(j*PIXELS+xMotionMesh.at<double>(i+1, j),
                                            (i+1)*PIXELS+yMotionMesh.at<double>(i+1, j)),
                                    Point2d((j+1)*PIXELS+xMotionMesh.at<double>(i, j+1),
                                            i*PIXELS+yMotionMesh.at<double>(i, j+1)),
                                    Point2d((j+1)*PIXELS+xMotionMesh.at<double>(i+1, j+1),
                                            (i+1)*PIXELS+yMotionMesh.at<double>(i+1, j+1))};
            Mat H = cv::findHomography(src, dst, cv::RANSAC);

            for (int k = PIXELS*i; k < PIXELS*(i+1); ++k) {
                for (int l = PIXELS*j; l < PIXELS*(j+1); ++l) {
                    double x = H.at<double>(0, 0)*l + H.at<double>(0, 1)*k + H.at<double>(0, 2);
                    double y = H.at<double>(1, 0)*l + H.at<double>(1, 1)*k + H.at<double>(1, 2);
                    double w = H.at<double>(2, 0)*l + H.at<double>(2, 1)*k + H.at<double>(2, 2);

                    if (w > 0) {
                        x = x/w;
                        y = y/w;
                    } else {
                        x = l;
                        y = k;
                    }

                    mapX.at<float>(k, l) = static_cast<float>(x);
                    //std::cout << mapX.at<float>(k, l) << std::endl;
                    mapY.at<float>(k, l) = static_cast<float>(y);
                }
            }
        }
    }

    // Repeat motion vectors for remaining frame in y-direction
    for (int i = PIXELS*xMotionMesh.rows; i < mapX.rows; ++i) {
        for (int k = 0; k < mapX.cols; ++k) {
            mapX.at<float>(i, k) = mapX.at<float>(PIXELS*xMotionMesh.rows-1, k);
            mapY.at<float>(i, k) = mapY.at<float>(PIXELS*xMotionMesh.rows-1, k);
        }
    }
    // Repeat motion vectors for remaining frame in x-direction
    for (int j = PIXELS*xMotionMesh.cols; j < mapX.cols; ++j) {
        for (int k = 0; k < mapX.rows; ++k) {
            mapX.at<float>(k, j) = mapX.at<float>(k, PIXELS*xMotionMesh.rows-1);
            mapY.at<float>(k, j) = mapY.at<float>(k, PIXELS*xMotionMesh.rows-1);
        }
    }

    cv::remap(frame, warpedFrame, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

}

}
