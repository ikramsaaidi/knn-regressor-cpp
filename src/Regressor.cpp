#include "Regressor.h"

double Regressor::score(const Matrix& X, const Matrix& y) {
    checkFitted();

    Matrix yPred = predict(X);
    int n = y.rows();

    double yMean = 0.0;
    for (int i = 0; i < n; ++i)
        yMean += y(i, 0);
    yMean /= static_cast<double>(n);

    double ssRes = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = y(i, 0) - yPred(i, 0);
        ssRes += diff * diff;
    }

    double ssTot = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = y(i, 0) - yMean;
        ssTot += diff * diff;
    }

    if (ssTot < 1e-12) return 1.0;

    return 1.0 - (ssRes / ssTot);
}