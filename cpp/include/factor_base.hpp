#pragma once
#include <Eigen/Dense>
#include <stdexcept>

class FactorBase {
public:
    virtual ~FactorBase() = default;

    // 核心计算接口
    virtual Eigen::VectorXd calculate(
        const Eigen::MatrixXd& data,
        const Eigen::VectorXd& params
    ) = 0;

    // 参数验证
    virtual bool validate_params(const Eigen::VectorXd& params) {
        // 默认实现 - 检查参数不为空
        if (params.size() == 0) {
            throw std::invalid_argument("参数不能为空");
        }
        return true;
    }

    // 通用工具方法
protected:
    double safe_divide(double numerator, double denominator) {
        if (denominator == 0.0) {
            return 0.0; // 避免除零错误
        }
        return numerator / denominator;
    }
};
