//#pragma once
//#include "factor_base.hpp"
//
//class CustomFactor : public FactorBase {
//public:
//    Eigen::VectorXd calculate(
//        const Eigen::MatrixXd& data,
//        const Eigen::VectorXd& params) override;
//};


#pragma once
#include "factor_base.hpp"
#include "market_data.hpp"
#include <Eigen/Dense>



class TR0 : public FactorBase {
public:
    TR0();

protected:
    // 实现信号生成
    std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        value_to_signal(const MarketData& market_data,
            const std::vector<int>& params) override;

    // 实现全量计算
    void eval_full(const MarketData& market_data,
        const std::vector<int>& params) override;

    // 实现增量更新
    void step_update(const MarketData& market_data,
        const std::vector<int>& params) override;

private:
    // RSI计算辅助函数
    void save_rsi_state(const MarketData& market_data, int period);
    double update_rsi(const MarketData& market_data, int period);

    // RSI计算状态
    double prev_avg_gain;
    double prev_avg_loss;
    MarketData price_window; // 使用MarketData存储价格窗口
};
