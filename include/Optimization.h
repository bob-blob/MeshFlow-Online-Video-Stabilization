#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;

using std::vector;

namespace meshflow {

class Optimizer
{
public:
    double gauss(int t, int r, int windowSize);

    /// Optimization methods solved by iterative Jacobi-based solver
    vector<Mat> offlinePathOptimization(const vector<Mat> cameraPath,
                                   int iterations,
                                   int windowSize);

    vector<Mat> onlinePathOptimization(const vector<Mat> cameraPath,
                                       int bufferSize,
                                       int iterations,
                                       int windowSize,
                                       int beta);
    // Implement parallel optimization

    Mat getCamPath(const vector<Mat>& camPath, int i, int j);
    Mat getCamPath(const vector<Mat>& camPath, int i, int j, int ub);
    Mat getCamPath(const vector<Mat>& camPath, int i, int j, int lb, int ub);


    Mat getW(const Mat& W, int t);
};

}
