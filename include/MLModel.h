#pragma once
#include "Matrix.h"
#include <stdexcept>

class MLModel {
public:
    virtual void   fit    (const Matrix& X, const Matrix& y) = 0;
    virtual Matrix predict(const Matrix& X)                  = 0;
    virtual double score  (const Matrix& X, const Matrix& y) = 0;

    virtual ~MLModel() {}
};