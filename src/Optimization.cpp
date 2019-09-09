#include "Optimization.h"

namespace meshflow {

double Optimizer::gauss(int t, int r, int windowSize) {
    if (abs(r - t) > windowSize)
        return 0;

    return exp( (-9 * pow((r - t), 2)) / pow(windowSize, 2));
}

/// Optimization methods solved by iterative Jacobi-based solver
void Optimizer::offlinePathOptimization(const vector<Mat>& cameraPath,
                                        vector<Mat>& p,
                                               int iterations,
                                               int windowSize) {
    double lambda_t = 100;
   // vector<Mat> p = cameraPath;
    p.reserve(cameraPath.size());
    for (int i = 0; i < cameraPath.size(); ++i) {
        p.push_back(Mat::zeros(cameraPath.back().rows, cameraPath.back().cols, CV_64F));
    }

    // Fill in the weight matrix
    Mat W = Mat::zeros(cameraPath.size(), cameraPath.size(), CV_64F);
    for (size_t t = 0; t < cameraPath.size(); ++t) {
        for (int r = -windowSize/2; r < windowSize/2 + 1; ++r) {
            if ( t + r < 0                 ||
                 t + r >= cameraPath.size() ||
                 r == 0 )
            {
                continue;
            }
            W.at<double>(t, t+r) = gauss(t, t+r, windowSize);
        }
    }

    // Calculate the gamma coefficients for each frame time
    Mat gamma = 1 + lambda_t * W * Mat::ones(cameraPath.size(), 1, CV_64F);

    for (int i = 0; i < cameraPath.back().rows; ++i) {
        for (int j = 0; j < cameraPath.back().cols; ++j) {
            Mat camPath = getCamPath(cameraPath, i, j); // Mat_<double>(1, t)
            Mat P;
            camPath.copyTo(P);
            for (int iter = 0; iter < iterations; ++iter) {
                P = (camPath + lambda_t * W * P) / gamma;
            }
            for (int k = 0; k < P.rows; ++k) {
                p[k].at<double>(i, j) = P.at<double>(k);
            }
        }
    }

}

vector<Mat> Optimizer::onlinePathOptimization(const vector<Mat>& cameraPath,
                                              int bufferSize,
                                              int iterations,
                                              int windowSize,
                                              int beta) {

    double lambda_t = 100;
    vector<Mat> p = cameraPath;

    /// Fill in the weights
    Mat W(bufferSize, bufferSize, CV_64F);
    for (int i = 0; i < W.rows; ++i) {
        for (int j = 0; j < W.cols; ++j) {
            W.at<double>(i, j) = gauss(i, j, windowSize);
        }
    }

    /// Iterative optimization process
    for (int i = 0; i < cameraPath.back().rows; ++i) {
        for (int j = 0; j < cameraPath.back().cols; ++j) {
            // y = []; d = None
            vector<double> y;
            Mat d, P;
            // Real-time optimization
            for (int t = 1; t < cameraPath.size()+1; ++t) {
                if (t < bufferSize + 1) {
                    Mat camPath = getCamPath(cameraPath, i, j, t);
                    camPath.copyTo(P);

                    if (!d.empty()) {
                        for (int k = 0; k < iterations; ++k) {
                            Mat Wt = getW(W, t);
                            Mat alpha = camPath + lambda_t * Wt * P;
                            Mat gamma = 1 + lambda_t * getW(W, t) * Mat::ones(t, 1, CV_64F);

                            for (int m = 0; m < alpha.cols-1; ++m) {
                                alpha.at<double>(m) += beta * d.at<double>(m);
                                gamma.at<double>(m) += beta;
                            }
                            P = alpha / gamma;
                        }
                    }
                } else {
                    Mat camPath = getCamPath(cameraPath, i, j, t-bufferSize, t);
                    camPath.copyTo(P);

                    for (int k = 0; k < iterations; ++k) {
                        Mat alpha = camPath + lambda_t * W * P;
                        Mat gamma = 1 + lambda_t * W * Mat::ones(bufferSize, 1, CV_64F);

                        for (int m = 0; m < alpha.cols-1; ++m) {
                            alpha.at<double>(m) += beta * d.at<double>(m+1);
                            gamma.at<double>(m) += beta;
                        }
                        P = alpha / gamma;
                    }
                }
                d = P;
                y.push_back(P.at<double>(P.rows-1));
            }
            for (int m = 0; m < y.size(); ++m) {
                p[m].at<double>(i, j) = y[m];
            }
        }
    }

    return p;
}

Mat Optimizer::getCamPath(const vector<Mat>& camPath, int i, int j) {
    Mat res(camPath.size(), 1, CV_64F);
    for (int k = 0; k < camPath.size(); ++k) {
        res.at<double>(k) = camPath[k].at<double>(i, j);
    }
    return res;
}

Mat Optimizer::getCamPath(const vector<Mat>& camPath, int i, int j, int ub) {
    Mat res(ub, 1, CV_64F);
    for (int k = 0; k < ub; ++k) {
        res.at<double>(k) = camPath[k].at<double>(i, j);
    }
    return res;
}

Mat Optimizer::getCamPath(const vector<Mat> &camPath, int i, int j, int lb, int ub) {
    Mat res(ub-lb, 1, CV_64F);

    for (int k = lb; k < ub; ++k) {
        res.at<double>(k-lb) = camPath[k].at<double>(i, j);
    }
    return res;
}

Mat Optimizer::getW(const Mat& W, int t) {
    Mat Wt(t, t, CV_64F);
    for (int i = 0; i < t; ++i) {
        for (int j = 0; j < t; ++j) {
            Wt.at<double>(i, j) = W.at<double>(i, j);
        }
    }
    return Wt;
}

}
