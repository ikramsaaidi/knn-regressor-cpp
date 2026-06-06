#include <iostream>
#include <iomanip>
#include <cmath>
#include "KNNRegressor.h"

// ─── Normalisation Z-score ───────────────────────────────
Matrix zScore(const Matrix& X) {
    int n = X.rows();
    int d = X.cols();
    Matrix result(n, d);

    for (int j = 0; j < d; ++j) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i)
            mean += X(i, j);
        mean /= static_cast<double>(n);

        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = X(i, j) - mean;
            var += diff * diff;
        }
        double stddev = std::sqrt(var / static_cast<double>(n));
        if (stddev < 1e-12) stddev = 1.0;

        for (int i = 0; i < n; ++i)
            result(i, j) = (X(i, j) - mean) / stddev;
    }
    return result;
}

int main() {
    std::cout << "=== KNN Regressor ===" << "\n\n";

    // ── Dataset : surface, nb pieces → prix (k€) ─────────
    int nTotal = 12;
    int nFeatures = 2;

    Matrix X(nTotal, nFeatures);
    Matrix y(nTotal, 1);

    double rawX[12][2] = {
        { 30, 1 }, { 45, 2 }, { 60, 2 }, { 75, 3 },
        { 90, 3 }, {105, 4 }, {120, 4 }, {150, 5 },
        { 50, 2 }, { 80, 3 }, { 95, 4 }, {110, 4 }
    };
    double rawY[12] = {
        120, 160, 200, 250,
        300, 360, 400, 480,
        175, 270, 320, 375
    };

    for (int i = 0; i < nTotal; ++i) {
        for (int j = 0; j < nFeatures; ++j)
            X(i, j) = rawX[i][j];
        y(i, 0) = rawY[i];
    }

    // ── Normalisation ─────────────────────────────────────
    Matrix Xn = zScore(X);

    // ── Split train/test (80/20) ──────────────────────────
    int nTrain = 9;
    int nTest  = nTotal - nTrain;

    Matrix X_train(nTrain, nFeatures);
    Matrix y_train(nTrain, 1);
    Matrix X_test (nTest,  nFeatures);
    Matrix y_test (nTest,  1);

    for (int i = 0; i < nTrain; ++i) {
        for (int j = 0; j < nFeatures; ++j)
            X_train(i, j) = Xn(i, j);
        y_train(i, 0) = y(i, 0);
    }
    for (int i = 0; i < nTest; ++i) {
        for (int j = 0; j < nFeatures; ++j)
            X_test(i, j) = Xn(nTrain + i, j);
        y_test(i, 0) = y(nTrain + i, 0);
    }

    // ── Entrainement ──────────────────────────────────────
    KNNRegressor model(3);
    model.fit(X_train, y_train);
    std::cout << "Modele entraine  K = " << model.getK() << "\n\n";

    // ── Predictions ───────────────────────────────────────
    Matrix pred = model.predict(X_test);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Resultats sur le jeu de test :\n";
    std::cout << "------------------------------------\n";
    std::cout << "  i  |  Reel (ke)  |  Predit (ke)\n";
    std::cout << "------------------------------------\n";
    for (int i = 0; i < nTest; ++i) {
        std::cout << "  " << i
                  << "  |   " << std::setw(7) << y_test(i, 0)
                  << "     |   " << std::setw(7) << pred(i, 0)
                  << "\n";
    }
    std::cout << "------------------------------------\n\n";

    // ── Score R2 ──────────────────────────────────────────
    double r2 = model.score(X_test, y_test);
    std::cout << "Score R2 = " << r2 << "\n\n";

    // ── Comparaison de K ──────────────────────────────────
    std::cout << "Influence du parametre K :\n";
    std::cout << "-------------------\n";
    std::cout << "  K  |     R2\n";
    std::cout << "-------------------\n";
    int kValues[5] = {1, 2, 3, 4, 5};
    for (int idx = 0; idx < 5; ++idx) {
        KNNRegressor m(kValues[idx]);
        m.fit(X_train, y_train);
        double s = m.score(X_test, y_test);
        std::cout << "  " << kValues[idx]
                  << "  |   " << s << "\n";
    }
    std::cout << "-------------------\n\n";

    // ── Test polymorphisme ────────────────────────────────
    std::cout << "Test polymorphisme (MLModel*) :\n";
    MLModel* ptr = new KNNRegressor(3);
    ptr->fit(X_train, y_train);
    Matrix res = ptr->predict(X_test);
    std::cout << "  Prediction point 0 : "
              << res(0, 0) << " ke\n";
    delete ptr;

    return 0;
}