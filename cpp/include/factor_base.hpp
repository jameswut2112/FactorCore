


#pragma once
#include "market_data.hpp"
#include <vector>
#include <tuple>
#include <stdexcept>
#include <Eigen/Dense>

class FactorBase {
public:
    FactorBase();
    virtual ~FactorBase() = default;

    // 参数配置接口
    void setParaRange(const std::vector<int>& pmin,
        const std::vector<int>& pmax,
        const std::vector<int>& pstep,
        const std::vector<int>& default_params);

    // 获取参数信息
    const std::vector<int>& get_pmin() const { return pmin; }
    const std::vector<int>& get_pmax() const { return pmax; }
    const std::vector<int>& get_pstep() const { return pstep; }
    const std::vector<int>& get_default() const { return default_params; }

    // 主计算接口：生成信号
    std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        eval(const MarketData& market_data,
            const std::vector<int>& params);

    // 获取当前因子值
    const Eigen::VectorXd& get_value() const { return value; }

protected:
    // 智能更新因子值（内部使用）
    void eval_smart(const MarketData& market_data,
        const std::vector<int>& params);

    // 信号生成虚函数
    virtual std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        value_to_signal(const MarketData& market_data,
            const std::vector<int>& params) = 0;

    // 全量计算（子类实现）
    virtual void eval_full(const MarketData& market_data,
        const std::vector<int>& params) = 0;

    // 增量更新（子类实现）
    virtual void step_update(const MarketData& market_data,
        const std::vector<int>& params) = 0;

    // 参数和状态
    std::vector<int> pmin;
    std::vector<int> pmax;
    std::vector<int> pstep;
    std::vector<int> default_params;
    Eigen::VectorXd value;
};
