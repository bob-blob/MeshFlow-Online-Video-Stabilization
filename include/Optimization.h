#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;

using std::vector;

namespace meshflow {

class Optimizer
{
public:
    static double gauss(int t, int r, int windowSize);

    /// Optimization methods solved by iterative Jacobi-based solver
    static void offlinePathOptimization(const vector<Mat>& cameraPath,
                                        vector<Mat>& p,
                                               int iterations = 100,
                                               int windowSize = 6);

    static vector<Mat> onlinePathOptimization(const vector<Mat>& cameraPath,
                                              int bufferSize = 200,
                                              int iterations = 10,
                                              int windowSize = 32,
                                              int beta = 1);
    // Implement parallel optimization

    static Mat getCamPath(const vector<Mat>& camPath, int i, int j);
    static Mat getCamPath(const vector<Mat>& camPath, int i, int j, int ub);
    static Mat getCamPath(const vector<Mat>& camPath, int i, int j, int lb, int ub);


    static Mat getW(const Mat& W, int t);
};

}
