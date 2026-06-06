#include "KNNRegressor.h"

// ─── Constructeur ────────────────────────────────────────
KNNRegressor::KNNRegressor(int k, DistanceMetric metric)
    : k_(k), metric_(metric)
{
    if (k <= 0)
        throw std::invalid_argument(
            "K doit etre un entier strictement positif.");
}

// ─── setK ────────────────────────────────────────────────
void KNNRegressor::setK(int k) {
    if (k <= 0)
        throw std::invalid_argument(
            "K doit etre un entier strictement positif.");
    k_ = k;
}

// ─── fit ─────────────────────────────────────────────────
void KNNRegressor::fit(const Matrix& X, const Matrix& y) {
    checkDimensions(X, y);

    if (k_ > X.rows())
        throw std::invalid_argument(
            "K ne peut pas depasser le nombre d'exemples.");

    X_train_ = X;
    y_train_ = y;
    isFitted_ = true;
}

// ─── computeDistance ─────────────────────────────────────
double KNNRegressor::computeDistance(const Matrix& A, int i,
                                     const Matrix& B, int j) const {
    double result = 0.0;
    int d = A.cols();

    if (metric_ == EUCLIDEAN) {
        for (int k = 0; k < d; ++k) {
            double diff = A(i, k) - B(j, k);
            result += diff * diff;
        }
        return std::sqrt(result);
    }

    for (int k = 0; k < d; ++k)
        result += std::abs(A(i, k) - B(j, k));

    return result;
}

// ─── predictOne ──────────────────────────────────────────
double KNNRegressor::predictOne(const Matrix& X, int rowIdx) const {
    int n = X_train_.rows();

    std::vector< std::pair<double, int> > dist(n);

    for (int i = 0; i < n; ++i)
        dist[i] = std::make_pair(
            computeDistance(X, rowIdx, X_train_, i), i);

    std::partial_sort(dist.begin(),
                      dist.begin() + k_,
                      dist.end());

    double sum = 0.0;
    for (int i = 0; i < k_; ++i)
        sum += y_train_(dist[i].second, 0);

    return sum / static_cast<double>(k_);
}

// ─── predict ─────────────────────────────────────────────
Matrix KNNRegressor::predict(const Matrix& X) {
    checkFitted();

    if (X.cols() != X_train_.cols())
        throw std::invalid_argument(
            "X doit avoir le meme nombre de features que X_train.");

    int m = X.rows();
    Matrix result(m, 1);

    for (int i = 0; i < m; ++i)
        result(i, 0) = predictOne(X, i);

    return result;
}

// ─── score ───────────────────────────────────────────────
double KNNRegressor::score(const Matrix& X, const Matrix& y) {
    return Regressor::score(X, y);
}