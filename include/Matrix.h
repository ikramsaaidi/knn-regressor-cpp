#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {
private:
    int rows_;
    int cols_;
    std::vector<double> data_;   // stockage 1D : data_[i*cols_ + j]

public:
    Matrix();
    Matrix(int rows, int cols, double val = 0.0);

    double&       operator()(int row, int col);
    double        operator()(int row, int col) const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    void print() const;
};