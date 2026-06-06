#include "Matrix.h"

Matrix::Matrix()
    : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols, double val)
    : rows_(rows), cols_(cols), data_(rows * cols, val) {}

double& Matrix::operator()(int row, int col) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index Matrix hors limites");
    return data_[row * cols_ + col];
}

double Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index Matrix hors limites");
    return data_[row * cols_ + col];
}

void Matrix::print() const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            std::cout << (*this)(i, j);
            if (j < cols_ - 1) std::cout << "\t";
        }
        std::cout << "\n";
    }
}