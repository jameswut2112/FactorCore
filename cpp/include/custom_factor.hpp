#pragma once
#include "factor_base.hpp"

class CustomFactor : public FactorBase {
public:
    Eigen::VectorXd calculate(
        const Eigen::MatrixXd& data,
        const Eigen::VectorXd& params) override;
};
