#pragma once
#include "MLModel.h"
#include <stdexcept>

class SupervisedModel : public MLModel {
protected:
    bool isFitted_;

    void checkFitted() const {
        if (!isFitted_)
            throw std::logic_error(
                "Appelez fit() avant predict() ou score().");
    }

    void checkDimensions(const Matrix& X, const Matrix& y) const {
        if (X.rows() != y.rows())
            throw std::invalid_argument(
                "X et y doivent avoir le meme nombre de lignes.");
        if (X.rows() == 0)
            throw std::invalid_argument(
                "Le dataset ne peut pas etre vide.");
    }

public:
    SupervisedModel() : isFitted_(false) {}
    virtual ~SupervisedModel() {}
};