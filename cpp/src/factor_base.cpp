//#include "factor_base.hpp"
//
//// �����ʵ�ֿ������գ���������Ҫ������߼�
//// ���麯������������ʵ��


#include "factor_base.hpp"

// ���캯��
FactorBase::FactorBase() {
    pmin = {};
    pmax = {};
    pstep = {};
    default_params = {};
}

// ���ò�����Χ
void FactorBase::setParaRange(const std::vector<int>& pmin,
    const std::vector<int>& pmax,
    const std::vector<int>& pstep,
    const std::vector<int>& default_params) {
    // ��֤������Χһ����
    if (pmin.size() != pmax.size() ||
        pmin.size() != pstep.size() ||
        pmin.size() != default_params.size()) {
        throw std::invalid_argument("All parameter arrays must have the same size");
    }

    // ��֤Ĭ��ֵ�ڷ�Χ��
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

// ���ܸ�������ֵ
void FactorBase::eval_smart(const MarketData& market_data,
    const std::vector<int>& params) {
    // ��֤�г�����
    market_data.validate();

    const int new_len = market_data.size();
    const int curr_len = value.size();

    // �����жϣ���ȫ����Python�߼���
    if (value.size() == 0 || new_len - curr_len >= 2) {
        // ����ȫ������
        eval_full(market_data, params);
    }
    else if (new_len - curr_len == 1) {
        // ִ����������
        step_update(market_data, params);
    }
    else if (new_len - curr_len == 0) {
        // ���ݺ�����ֵ�ȳ������ü���
        return;
    }
    else {
        // �쳣����������ݳ���С�ڵ�ǰ����
        throw std::runtime_error("Invalid data length difference: " +
            std::to_string(new_len - curr_len));
    }
}

// ������ӿ�
std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
FactorBase::eval(const MarketData& market_data,
    const std::vector<int>& params) {
    // 1. ��֤�г�����
    market_data.validate();

    // 2. ���û���ṩ������ʹ��Ĭ��ֵ
    std::vector<int> actual_params = params;
    if (actual_params.empty()) {
        actual_params = default_params;
    }

    // 3. ��֤��������
    if (actual_params.size() != pmin.size()) {
        throw std::invalid_argument("Parameter count mismatch. Expected " +
            std::to_string(pmin.size()) +
            ", got " +
            std::to_string(actual_params.size()));
    }

    // 4. ��֤������Χ
    for (size_t i = 0; i < actual_params.size(); ++i) {
        if (actual_params[i] < pmin[i] || actual_params[i] > pmax[i]) {
            throw std::out_of_range("Parameter " + std::to_string(i) +
                " out of range [" +
                std::to_string(pmin[i]) + ", " +
                std::to_string(pmax[i]) + "]");
        }
    }

    // 5. ���ܸ�������ֵ
    eval_smart(market_data, actual_params);

    // 6. ���ɽ����ź�
    return value_to_signal(market_data, actual_params);
}
