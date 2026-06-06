#pragma once
#include "SupervisedModel.h"
#include <cmath>

class Regressor : public SupervisedModel {
public:
    virtual ~Regressor() {}

    double score(const Matrix& X, const Matrix& y) override;
};