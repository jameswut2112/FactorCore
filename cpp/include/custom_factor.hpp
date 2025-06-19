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
    // ʵ���ź�����
    std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        value_to_signal(const MarketData& market_data,
            const std::vector<int>& params) override;

    // ʵ��ȫ������
    void eval_full(const MarketData& market_data,
        const std::vector<int>& params) override;

    // ʵ����������
    void step_update(const MarketData& market_data,
        const std::vector<int>& params) override;

private:
    // RSI���㸨������
    void save_rsi_state(const MarketData& market_data, int period);
    double update_rsi(const MarketData& market_data, int period);

    // RSI����״̬
    double prev_avg_gain;
    double prev_avg_loss;
    MarketData price_window; // ʹ��MarketData�洢�۸񴰿�
};
