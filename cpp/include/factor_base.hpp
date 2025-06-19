


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

    // �������ýӿ�
    void setParaRange(const std::vector<int>& pmin,
        const std::vector<int>& pmax,
        const std::vector<int>& pstep,
        const std::vector<int>& default_params);

    // ��ȡ������Ϣ
    const std::vector<int>& get_pmin() const { return pmin; }
    const std::vector<int>& get_pmax() const { return pmax; }
    const std::vector<int>& get_pstep() const { return pstep; }
    const std::vector<int>& get_default() const { return default_params; }

    // ������ӿڣ������ź�
    std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        eval(const MarketData& market_data,
            const std::vector<int>& params);

    // ��ȡ��ǰ����ֵ
    const Eigen::VectorXd& get_value() const { return value; }

protected:
    // ���ܸ�������ֵ���ڲ�ʹ�ã�
    void eval_smart(const MarketData& market_data,
        const std::vector<int>& params);

    // �ź������麯��
    virtual std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
        value_to_signal(const MarketData& market_data,
            const std::vector<int>& params) = 0;

    // ȫ�����㣨����ʵ�֣�
    virtual void eval_full(const MarketData& market_data,
        const std::vector<int>& params) = 0;

    // �������£�����ʵ�֣�
    virtual void step_update(const MarketData& market_data,
        const std::vector<int>& params) = 0;

    // ������״̬
    std::vector<int> pmin;
    std::vector<int> pmax;
    std::vector<int> pstep;
    std::vector<int> default_params;
    Eigen::VectorXd value;
};
