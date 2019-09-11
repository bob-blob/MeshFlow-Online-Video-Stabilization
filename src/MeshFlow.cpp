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
                               Mat& xMotionMesh, Mat& yMotionMesh,
                               vector<std::pair<double, double>>& lambdas,
                               cv::Size sz) {

    int cols = static_cast<int>(oldFrame.cols / sz.width + 1);
    int rows = static_cast<int>(oldFrame.rows / sz.height + 1);

    Mat H = cv::findHomography(oldPoints, newPoints, cv::RANSAC);

    /// Find information for offline adaptive lambda
    /// TODO: FIX THIS
    double translationalElement = sqrt(pow(H.at<double>(0, 2), 2) + pow(H.at<double>(1, 2), 2));
    double l1 = -1.93 * translationalElement + 0.95;

    Mat affinePart = H(cv::Range(0, 2), cv::Range(0, 2));
    Mat eigenvalues;
    cv::eigen(affinePart, eigenvalues);
    double affineComponent = eigenvalues.at<double>(0) / eigenvalues.at<double>(1);
    double l2 = 5.83 * affineComponent + 4.88;
    lambdas.push_back(std::make_pair(l1, l2));

    Map xMotion, yMotion;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Point2d point(sz.width*j, sz.height*i);
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
            Point2d vertex(sz.width*j, sz.height*i);
            for (size_t k = 0; k < oldPoints.size(); ++k) {
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
    Map::iterator it = xMotion.begin();
    while (it != xMotion.end()) {

        Map::iterator itX = temp_xMotion.find(it->first);
        if (itX != temp_xMotion.end())
        {
            std::sort(itX->second.begin(), itX->second.end());
            xMotionMesh.at<double>(it->first.first, it->first.second) = xMotion[it->first].back() + temp_xMotion[it->first].at(temp_xMotion[it->first].size()/2);
        } else {
            xMotionMesh.at<double>(it->first.first, it->first.second) = xMotion[it->first].back();
        }

        Map::iterator itY = temp_yMotion.find(it->first);
        if (itY != temp_yMotion.end())
        {
            std::sort(itY->second.begin(), itY->second.end());
            yMotionMesh.at<double>(it->first.first, it->first.second) = yMotion[it->first].back() + temp_yMotion[it->first].at(temp_yMotion[it->first].size()/2);
        } else {
            yMotionMesh.at<double>(it->first.first, it->first.second) = yMotion[it->first].back();
        }

        it++;
    }

    // TODO: Apply the second medial filter over the motion field to discard the outliers
    // median kernel [3x3] with Replicate borders

    xMotionMesh.convertTo(xMotionMesh, CV_32F);
    cv::medianBlur(xMotionMesh, xMotionMesh, 3);
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
                             Mat& warpedFrame, cv::Size sz) {

    Mat mapX = Mat::zeros(frame.rows, frame.cols, CV_32F);
    Mat mapY = Mat::zeros(frame.rows, frame.cols, CV_32F);

    for (int i = 0; i < xMotionMesh.rows-1; ++i) {
        for (int j = 0; j < xMotionMesh.cols-1; ++j) {

            vector<Point2d> src = { Point2d(j*sz.width, i*sz.height),
                                    Point2d(j*sz.width, (i+1)*sz.height),
                                    Point2d((j+1)*sz.width, i*sz.height),
                                    Point2d((j+1)*sz.width, (i+1)*sz.height)};

            vector<Point2d> dst = { Point2d(j* sz.width+xMotionMesh.at<double>(i, j),
                                    i*sz.height+yMotionMesh.at<double>(i, j)),
                                    Point2d(j*sz.width+xMotionMesh.at<double>(i+1, j),
                                    (i+1)*sz.height+yMotionMesh.at<double>(i+1, j)),
                                    Point2d((j+1)*sz.width+xMotionMesh.at<double>(i, j+1),
                                    i*sz.height+yMotionMesh.at<double>(i, j+1)),
                                    Point2d((j+1)*sz.width+ xMotionMesh.at<double>(i+1, j+1),
                                    (i+1)*sz.height+yMotionMesh.at<double>(i+1, j+1)) };

            Mat H = cv::findHomography(src, dst, cv::RANSAC);

            for (int k = sz.height*i; k < sz.height*(i+1); ++k) {
                for (int l = sz.width*j; l < sz.width*(j+1); ++l) {
                    double x = H.at<double>(0, 0)*l + H.at<double>(0, 1)*k + H.at<double>(0, 2);
                    double y = H.at<double>(1, 0)*l + H.at<double>(1, 1)*k + H.at<double>(1, 2);
                    double w = H.at<double>(2, 0)*l + H.at<double>(2, 1)*k + H.at<double>(2, 2);

                    if (w != 0) { // ._. this produces correct results
                        x = x/w;
                        y = y/w;
                    } else {
                        x = l;
                        y = k;
                    }

                    mapX.at<float>(k, l) = static_cast<float>(x);
                    mapY.at<float>(k, l) = static_cast<float>(y);
                }
            }
        }
    }

    cv::remap(frame, warpedFrame, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

}

}
