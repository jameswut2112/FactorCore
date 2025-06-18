#pragma once
#include <Eigen/Dense>
#include <stdexcept>

class FactorBase {
public:
    virtual ~FactorBase() = default;

    // ���ļ���ӿ�
    virtual Eigen::VectorXd calculate(
        const Eigen::MatrixXd& data,
        const Eigen::VectorXd& params
    ) = 0;

    // ������֤
    virtual bool validate_params(const Eigen::VectorXd& params) {
        // Ĭ��ʵ�� - ��������Ϊ��
        if (params.size() == 0) {
            throw std::invalid_argument("��������Ϊ��");
        }
        return true;
    }

    // ͨ�ù��߷���
protected:
    double safe_divide(double numerator, double denominator) {
        if (denominator == 0.0) {
            return 0.0; // ����������
        }
        return numerator / denominator;
    }
};
