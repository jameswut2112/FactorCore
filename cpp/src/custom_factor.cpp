#include "custom_factor.hpp"
#include <cmath>

Eigen::VectorXd CustomFactor::calculate(
    const Eigen::MatrixXd& data,
    const Eigen::VectorXd& params)
{
    // 示例计算：加权移动平均
    const int n = data.rows();
    const int window = static_cast<int>(params[0]);

    if (window <= 0 || window > n) {
        throw std::invalid_argument("无效的窗口大小");
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

    for (int i = window - 1; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < window; ++j) {
            sum += data(i - j, 0); // 假设使用第一列
        }
        result(i) = sum / window;
    }

    return result;
}
