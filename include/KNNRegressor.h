#pragma once
#include "Regressor.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <stdexcept>

enum DistanceMetric {
    EUCLIDEAN,
    MANHATTAN
};

class KNNRegressor : public Regressor {
private:
    int            k_;
    DistanceMetric metric_;
    Matrix         X_train_;
    Matrix         y_train_;

    double computeDistance(const Matrix& A, int i,
                           const Matrix& B, int j) const;

    double predictOne(const Matrix& X, int rowIdx) const;

public:
    explicit KNNRegressor(int k = 5,
                          DistanceMetric metric = EUCLIDEAN);

    ~KNNRegressor() {}

    void   fit    (const Matrix& X, const Matrix& y) override;
    Matrix predict(const Matrix& X)                  override;
    double score  (const Matrix& X, const Matrix& y) override;

    int  getK() const { return k_; }
    void setK(int k);
};