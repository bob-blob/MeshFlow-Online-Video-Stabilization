#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <opencv2/opencv.hpp>

#include "MeshFlow.h"

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
                                        vector<std::pair<double, double>>& lambdas,
                                        int iterations = 100,
                                        int windowSize = 6);

    static void onlinePathOptimization(const vector<Mat>& cameraPath,
                                              vector<Mat>& optimizedPath,
                                              int bufferSize = 200,
                                              int iterations = 10,
                                              int windowSize = 32,
                                              int beta = 1);

    static void offlinePathOptimizationLemon(const vector<Mat>& cameraPath,
                                             vector<Mat>& optimizedPath);
    // Implement parallel optimization

    static Mat getCamPath(const vector<Mat>& camPath, int i, int j);
    static Mat getCamPath(const vector<Mat>& camPath, int i, int j, int ub);
    static Mat getCamPath(const vector<Mat>& camPath, int i, int j, int lb, int ub);


    static Mat getW(const Mat& W, int t);
};

}

#endif
