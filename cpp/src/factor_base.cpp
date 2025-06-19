//#include "factor_base.hpp"
//
//// 基类的实现可以留空，除非有需要共享的逻辑
//// 纯虚函数在派生类中实现


#include "factor_base.hpp"

// 构造函数
FactorBase::FactorBase() {
    pmin = {};
    pmax = {};
    pstep = {};
    default_params = {};
}

// 设置参数范围
void FactorBase::setParaRange(const std::vector<int>& pmin,
    const std::vector<int>& pmax,
    const std::vector<int>& pstep,
    const std::vector<int>& default_params) {
    // 验证参数范围一致性
    if (pmin.size() != pmax.size() ||
        pmin.size() != pstep.size() ||
        pmin.size() != default_params.size()) {
        throw std::invalid_argument("All parameter arrays must have the same size");
    }

    // 验证默认值在范围内
    for (size_t i = 0; i < default_params.size(); ++i) {
        if (default_params[i] < pmin[i] || default_params[i] > pmax[i]) {
            throw std::out_of_range("Default parameter " + std::to_string(i) +
                " out of range [" +
                std::to_string(pmin[i]) + ", " +
                std::to_string(pmax[i]) + "]");
        }
    }

    this->pmin = pmin;
    this->pmax = pmax;
    this->pstep = pstep;
    this->default_params = default_params;
}

// 智能更新因子值
void FactorBase::eval_smart(const MarketData& market_data,
    const std::vector<int>& params) {
    // 验证市场数据
    market_data.validate();

    const int new_len = market_data.size();
    const int curr_len = value.size();

    // 场景判断（完全按照Python逻辑）
    if (value.size() == 0 || new_len - curr_len >= 2) {
        // 触发全量计算
        eval_full(market_data, params);
    }
    else if (new_len - curr_len == 1) {
        // 执行增量更新
        step_update(market_data, params);
    }
    else if (new_len - curr_len == 0) {
        // 数据和因子值等长，不用计算
        return;
    }
    else {
        // 异常情况：新数据长度小于当前长度
        throw std::runtime_error("Invalid data length difference: " +
            std::to_string(new_len - curr_len));
    }
}

// 主计算接口
std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
FactorBase::eval(const MarketData& market_data,
    const std::vector<int>& params) {
    // 1. 验证市场数据
    market_data.validate();

    // 2. 如果没有提供参数，使用默认值
    std::vector<int> actual_params = params;
    if (actual_params.empty()) {
        actual_params = default_params;
    }

    // 3. 验证参数个数
    if (actual_params.size() != pmin.size()) {
        throw std::invalid_argument("Parameter count mismatch. Expected " +
            std::to_string(pmin.size()) +
            ", got " +
            std::to_string(actual_params.size()));
    }

    // 4. 验证参数范围
    for (size_t i = 0; i < actual_params.size(); ++i) {
        if (actual_params[i] < pmin[i] || actual_params[i] > pmax[i]) {
            throw std::out_of_range("Parameter " + std::to_string(i) +
                " out of range [" +
                std::to_string(pmin[i]) + ", " +
                std::to_string(pmax[i]) + "]");
        }
    }

    // 5. 智能更新因子值
    eval_smart(market_data, actual_params);

    // 6. 生成交易信号
    return value_to_signal(market_data, actual_params);
}
