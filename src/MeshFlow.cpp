#include "MeshFlow.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace meshflow {

using cv::Vec2d;
using cv::Mat;
using cv::Point2d;
using std::vector;

Mesh::Mesh(cv::Size shape)
    : colNum{shape.width},
      rowNum{shape.height}
{}

std::vector<double> Mesh::getPropagationVector(int row, int col) {
    return vector<double>(meshMotionData[this->rows() * row + col].begin() + 1,
                          meshMotionData[this->rows() * row + col].end());
}

double Mesh::getInitialMotion(int row, int col) const {
    return meshMotionData[this->rows() * row + col][0];
}

void Mesh::setInitialMotion(int row, int col, double motion) {
    meshMotionData[this->rows() * row + col][0] = motion;
}

void Mesh::sortPropagationVectors() {
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < this->cols(); ++j) {
            std::sort(meshMotionData[this->rows() * i + j].begin()+1,
                      meshMotionData[this->rows() * i + j].end());
        }
    }
}

int Mesh::cols() const {
    return colNum;
}

int Mesh::rows() const {
    return rowNum;
}

MeshFlow::MeshFlow() {}

MeshFlow::MeshFlow(const config::MeshFlowConfiguration& config)
    : motionPropagationRadius(config.motionPropagationRadius) {

}

void MeshFlow::motionPropagate(const vector<Point2d>& oldPoints,
                               const vector<Point2d>& newPoints,
                               const Mat& oldFrame,
                               Mat& xMotionMesh, Mat& yMotionMesh,
                               vector<std::pair<double, double>>& lambdas,
                               cv::Size gridSize) {

    int cols = static_cast<int>(oldFrame.cols / gridSize.width + 1); // number of column vertices
    int rows = static_cast<int>(oldFrame.rows / gridSize.height + 1); // number of row vertices
    cv::Size meshSize(cols, rows);

    /// Find adaptive lambdas using global homography
    Mat H = cv::findHomography(oldPoints, newPoints);
    double lambda1 = translationalElement(H, cv::Size(oldFrame.cols, oldFrame.rows)),
           lambda2 = affineComponent(H);
    lambdas.push_back(std::make_pair(lambda1, lambda2));

    Map xMotionInitial, yMotionInitial, xMotionFilled, yMotionFilled;
    initializeMotionMesh(H, meshSize, gridSize, xMotionInitial, yMotionInitial);
    propogateFeatureMotion(oldPoints, newPoints, H, meshSize, gridSize, xMotionFilled, yMotionFilled);

    // Apply first median filter on obtained motion for each vertex
    sparseMedianFilter(xMotionInitial, xMotionFilled, xMotionMesh);
    sparseMedianFilter(yMotionInitial, yMotionFilled, yMotionMesh);

    // Apply the second medial filter over the motion field to discard the outliers. median kernel [3x3] with Replicate borde
    spatialMedianFilter(xMotionMesh, 3);
    spatialMedianFilter(yMotionMesh, 3);
}

void MeshFlow::generateVertexProfiles(const Mat& xMotionMesh,
                                      const Mat& yMotionMesh,
                                      vector<Mat>& xPaths, vector<Mat>& yPaths) {
    xPaths.push_back(xMotionMesh + xPaths.back());
    yPaths.push_back(yMotionMesh + yPaths.back());
}

void MeshFlow::meshWarpFrame(const Mat& frame, const Mat& xMotionMesh, const Mat& yMotionMesh, Mat& warpedFrame, cv::Size meshCellSize) {

    Mat mapX = Mat::zeros(frame.rows, frame.cols, CV_32F);
    Mat mapY = Mat::zeros(frame.rows, frame.cols, CV_32F);

    for (int i = 0; i < xMotionMesh.rows-1; ++i) {
        for (int j = 0; j < xMotionMesh.cols-1; ++j) {

            vector<Point2d> src = { Point2d(j*meshCellSize.width, i*meshCellSize.height),
                                    Point2d(j*meshCellSize.width, (i+1)*meshCellSize.height),
                                    Point2d((j+1)*meshCellSize.width, i*meshCellSize.height),
                                    Point2d((j+1)*meshCellSize.width, (i+1)*meshCellSize.height)};

            vector<Point2d> dst = { Point2d(j*meshCellSize.width+xMotionMesh.at<double>(i, j),
                                    i*meshCellSize.height+yMotionMesh.at<double>(i, j)),
                                    Point2d(j*meshCellSize.width+xMotionMesh.at<double>(i+1, j),
                                    (i+1)*meshCellSize.height+yMotionMesh.at<double>(i+1, j)),
                                    Point2d((j+1)*meshCellSize.width+xMotionMesh.at<double>(i, j+1),
                                    i*meshCellSize.height+yMotionMesh.at<double>(i, j+1)),
                                    Point2d((j+1)*meshCellSize.width+ xMotionMesh.at<double>(i+1, j+1),
                                    (i+1)*meshCellSize.height+yMotionMesh.at<double>(i+1, j+1)) };

            Mat H = cv::findHomography(src, dst, 0);

            // Transform indexes for image cell and build the map for remap
            for (int yCellIndex = meshCellSize.height*i; yCellIndex < meshCellSize.height*(i+1); ++yCellIndex) {
                for (int xCellIndex = meshCellSize.width*j; xCellIndex < meshCellSize.width*(j+1); ++xCellIndex) {
                    double X = H.at<double>(0, 0) * xCellIndex + H.at<double>(0, 1) * yCellIndex + H.at<double>(0, 2);
                    double Y = H.at<double>(1, 0) * xCellIndex + H.at<double>(1, 1) * yCellIndex + H.at<double>(1, 2);
                    double W = H.at<double>(2, 0) * xCellIndex + H.at<double>(2, 1) * yCellIndex + H.at<double>(2, 2);

                    W = W ? 1.0 / W : 0;
                    mapX.at<float>(yCellIndex, xCellIndex) = X * W;
                    mapY.at<float>(yCellIndex, xCellIndex) = Y * W;
                }
            }
        }
    }
    cv::remap(frame, warpedFrame, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

void MeshFlow::initializeMotionMesh(const Mat& homography, cv::Size meshSize, cv::Size gridCellSize, Map& xMotion, Map& yMotion) {
    // Prewarp. Ensures that each mesh vertex atleast contains global motion
    for (int i = 0; i < meshSize.height; ++i) {
        for (int j = 0; j < meshSize.width; ++j) {
            Point2d point(gridCellSize.width*j, gridCellSize.height*i);
            Point2d pointTransform = transformPoint(homography, point);

            xMotion[std::make_pair(i, j)].push_back(point.x - pointTransform.x);
            yMotion[std::make_pair(i, j)].push_back(point.y - pointTransform.y);
        }
    }
}

void MeshFlow::propogateFeatureMotion(const vector<Point2d>& oldFeatures, const vector<Point2d>& newFeatures, const Mat& homography, cv::Size meshSize, cv::Size gridCellSize, Map& xMotion, Map& yMotion) {
    for (int i = 0; i < meshSize.height; ++i) {
        for (int j = 0; j < meshSize.width; ++j) {
            Point2d vertex(gridCellSize.width*j, gridCellSize.height*i);

            for (size_t k = 0; k < oldFeatures.size(); ++k) {
                double distance = sqrt(pow(vertex.x - oldFeatures[k].x, 2) + pow(vertex.y - oldFeatures[k].y, 2));
                //double inEllipse = pow(vertex.x - oldFeatures[k].x, 2)/pow(RADIUS_X, 2) + pow(vertex.y - oldFeatures[k].y, 2)/pow(RADIUS_Y, 2);

                if(distance < motionPropagationRadius) { // Check if feature lies in in the radius of a circle then propagate motion to vertex
                //if (inEllipse <= 1) {
                    Point2d pointTransform = transformPoint(homography, oldFeatures[k]);

                    xMotion[std::make_pair(i, j)].push_back(newFeatures[k].x - pointTransform.x);
                    yMotion[std::make_pair(i, j)].push_back(newFeatures[k].y - pointTransform.y);
                }
            }
        }
    }
}

void MeshFlow::sparseMedianFilter(Map& initialMotionField, Map& propagatedMotionField, Mat& filteredMotionField) {
    Map::iterator it = initialMotionField.begin();
    while (it != initialMotionField.end()) {

        std::pair<int, int> vertexIndex = it->first;
        Map::iterator itVertexVector = propagatedMotionField.find(vertexIndex);

        if (itVertexVector != propagatedMotionField.end())
        {
            std::sort(itVertexVector->second.begin(), itVertexVector->second.end());
            filteredMotionField.at<double>(vertexIndex.first, vertexIndex.second) = initialMotionField[vertexIndex].back() + propagatedMotionField[vertexIndex].at(propagatedMotionField[vertexIndex].size()/2);
        } else {
            filteredMotionField.at<double>(vertexIndex.first, vertexIndex.second) = initialMotionField[vertexIndex].back();
        }

        it++;
    }
}

void MeshFlow::spatialMedianFilter(Mat& motionField, int kernelSize) {
    // Second median filter
    motionField.convertTo(motionField, CV_32F);
    cv::medianBlur(motionField, motionField, kernelSize);
    motionField.convertTo(motionField, CV_64F);
}

double MeshFlow::translationalElement(const Mat& homography, cv::Size frameResolution) {
    double intermResult = sqrt(
                pow(homography.at<double>(0, 2) / frameResolution.width, 2) +
                pow(homography.at<double>(1, 2) / frameResolution.height, 2));

    return -1.93 * intermResult + 0.95;     // Refer to the paper to find out about these empirically extracted coefficient
}

double MeshFlow::affineComponent(const Mat& homography) {
    Mat affinePart = homography(cv::Range(0, 2), cv::Range(0, 2)),
        eigenvalues;
    cv::eigen(affinePart, eigenvalues);

    double intermResult = eigenvalues.at<double>(0) / eigenvalues.at<double>(1);

    return 5.83 * intermResult - 4.83;      // In paper it is (+ 4.83)
}

Point2d MeshFlow::transformPoint(const Mat& H, const Point2d& point) {

    double a = H.at<double>(0, 0) * point.x + H.at<double>(0, 1) * point.y + H.at<double>(0, 2);
    double b = H.at<double>(1, 0) * point.x + H.at<double>(1, 1) * point.y + H.at<double>(1, 2);
    double c = H.at<double>(2, 0) * point.x + H.at<double>(2, 1) * point.y + H.at<double>(2, 2);

    return Point2d(a/c, b/c);
}

}
