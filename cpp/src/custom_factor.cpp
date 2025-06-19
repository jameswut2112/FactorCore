//#include "custom_factor.hpp"
//#include <cmath>
//
//Eigen::VectorXd CustomFactor::calculate(
//    const Eigen::MatrixXd& data,
//    const Eigen::VectorXd& params)
//{
//    // ʾ�����㣺��Ȩ�ƶ�ƽ��
//    const int n = data.rows();
//    const int window = static_cast<int>(params[0]);
//
//    if (window <= 0 || window > n) {
//        throw std::invalid_argument("��Ч�Ĵ��ڴ�С");
//    }
//
//    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
//
//    for (int i = window - 1; i < n; ++i) {
//        double sum = 0.0;
//        for (int j = 0; j < window; ++j) {
//            sum += data(i - j, 0); // ����ʹ�õ�һ��
//        }
//        result(i) = sum / window;
//    }
//
//    return result;
//}



#include "custom_factor.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>
#include "elutils.hpp"

TR0::TR0() {
    // ����Ĭ�ϲ�����Χ
    setParaRange(
        { 1, 1 },     // pmin
        { 200, 100 }, // pmax
        { 1, 1 },     // pstep
        { 15, 60 }    // default_params
    );

    // ��ʼ��״̬����
    prev_avg_gain = 0.0;
    prev_avg_loss = 0.0;
}

// ȫ����������ֵ
void TR0::eval_full(const MarketData& market_data,
    const std::vector<int>& params) {
    /*const int period = std::max(2, params[0]);
    const int n = market_data.close.size();*/

    if (params.empty() || params[0] <= 1) {
        throw std::invalid_argument("Invalid RSI period parameter");
    }

    const int period = params[0];
    const int n = market_data.close.size();

	// ���ݳ�����֤��ȷ�����㹻�����ݵ����RSI����
    if (n < period + 1) {
        value.resize(n);
        value.setConstant(std::numeric_limits<double>::quiet_NaN());
        std::cerr << "[WARNING] Insufficient data for RSI calculation. Need "
            << period + 1 << " points, got " << n << std::endl;
        return;
    }


    // ��ȫ����delta_size
    const int delta_size = n - 1;
    if (delta_size <= 0 || delta_size >= market_data.close.size()) {
        throw std::runtime_error("Invalid delta calculation");
    }


    try {
        // ʹ��segment���head/tail����ȫ
        const Eigen::VectorXd deltas = market_data.close.segment(1, delta_size) -
            market_data.close.segment(0, delta_size);

        // �����������ʧ
        const Eigen::VectorXd gains = deltas.cwiseMax(0.0);
        const Eigen::VectorXd losses = (-deltas).cwiseMax(0.0);

        // ��ʼ��RSIֵ
        value.resize(n);
        value.setZero();

        // �����ʼƽ���������ʧ��ǰperiod���㣩
        double avg_gain = gains.head(period).mean();
        double avg_loss = losses.head(period).mean();

        // ����ǰperiod+1���RSI����period�㣩
        if (avg_loss < 1e-9) {
            value(period) = (avg_gain < 1e-9) ? 50.0 : 100.0;
        }
        else {
            const double rs = avg_gain / avg_loss;
            value(period) = 100.0 - (100.0 / (1.0 + rs));
        }

        // ����������RSI
        for (int i = period + 1; i < n; ++i) {
            // ����ƽ���������ʧ��ʹ��Wilderƽ����
            avg_gain = (avg_gain * (period - 1) + gains(i - 1)) / period;
            avg_loss = (avg_loss * (period - 1) + losses(i - 1)) / period;

            // ����RS
            const double rs = (avg_loss < 1e-9) ?
                std::numeric_limits<double>::infinity() :
                avg_gain / avg_loss;

            // ����RSI
            value(i) = 100.0 - (100.0 / (1.0 + rs));
        }

        // ... ��������߼����ֲ��� ...
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] in eval_full: " << e.what() << std::endl;
        throw;
    }


    // ����״̬������������
    save_rsi_state(market_data, period);
}

// ������������ֵ
void TR0::step_update(const MarketData& market_data,
    const std::vector<int>& params) {
    const int new_size = market_data.size();
    const int current_size = value.size();

    // �ϸ���֤���ݳ��Ȳ�
    if (new_size <= current_size) {
        throw std::runtime_error("New data size must be greater than current size");
    }

    if (new_size - current_size > 1) {
        // ������ݲ��̫�󣬻��˵�ȫ������
        eval_full(market_data, params);
        return;
    }

    // ִ����������
    const double new_rsi = update_rsi(market_data, params[0]);

    // ��չvalue����
    Eigen::VectorXd new_value(current_size + 1);
    new_value << value, new_rsi;
    value = new_value;
}

// ���ɽ����ź�
std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
TR0::value_to_signal(const MarketData& market_data,
    const std::vector<int>& params) {
    // ���ֵδ���㣬�Ƚ���ȫ������
    if (value.size() == 0) {
        eval_full(market_data, params);
    }

    const double threshold = 50.0 + 0.5 * params[1];
    const int n = value.size();

    // ʹ��Eigen�����������������ź�
    Eigen::VectorXi buy_signal = (value.array() > threshold).cast<int>();
    Eigen::VectorXi sell_signal = (value.array() < (100.0 - threshold)).cast<int>();
    Eigen::VectorXi close_signal = Eigen::VectorXi::Zero(n);

    return std::make_tuple(buy_signal, sell_signal, close_signal);
}

//// ����RSI����״̬
//void TR0::save_rsi_state(const MarketData& market_data, int period) {
//    const int n = market_data.size();
//
//    if (n < period + 1) {
//        // ���ݲ��㣬����ȫ��
//        price_window = market_data;
//        return;
//    }
//
//    // �������period+1���۸��
//    price_window = market_data.slice(n - period - 1, n);
//}

void TR0::save_rsi_state(const MarketData& market_data, int period) {
    const int n = market_data.close.size();

    if (n == 0) {
        price_window = MarketData(); // ����Ϊ��
        return;
    }

    // ���㰲ȫ��Χ
    const int start_idx = std::max(0, n - period - 1);
    const int length = n - start_idx;

    // ʹ�ð�ȫ��Ƭ����
    price_window = market_data.slice(start_idx, n);
}


// ����RSIֵ��������
double TR0::update_rsi(const MarketData& market_data, int period) {

    // ������֤
    if (period <= 1) {
        throw std::invalid_argument("RSI period must be greater than 1");
    }

    // Replace the usage of `back()` with the appropriate Eigen method to access the last element.  
    // ��ȫ��ȡ���һ�����̼�
    const double new_close = market_data.close[market_data.close.size() - 1];
    //const double new_close = market_data.close.size() > 0 ? market_data.close(market_data.close.size() - 1) : 0.0;

    // ���¼۸񴰿�
    if (price_window.size() == 0) {
        // �״θ��£���ʼ������
        price_window = market_data;
        return 50.0; // Ĭ��RSIֵ
    }

    try {
        // �����´��ڸ���
        MarketData new_window = price_window;

        // ��������ݵ�
        new_window.close.conservativeResize(new_window.close.size() + 1);
        new_window.close.tail(1) << new_close;

        // �޼����ڴ�С
        if (new_window.close.size() > period + 1) {
            new_window.close = new_window.close.tail(period + 1);
        }

        price_window = new_window;

        // ����Ƿ����㹻���ݼ���
        if (price_window.close.size() < 2) {
            return 50.0;
        }

        // ��������delta
        const double delta = price_window.close[price_window.close.size() - 1] -
            price_window.close[price_window.close.size() - 2];

        const double gain = std::max(delta, 0.0);
        const double loss = std::max(-delta, 0.0);

        // ����ƽ���������ʧ
        prev_avg_gain = (prev_avg_gain * (period - 1) + gain) / period;
        prev_avg_loss = (prev_avg_loss * (period - 1) + loss) / period;

        // ����������
        if (prev_avg_loss < 1e-9) {
            return (prev_avg_gain < 1e-9) ? 50.0 : 100.0;
        }

        // ����RS
        const double rs = prev_avg_gain / prev_avg_loss;

        // ��������RSIֵ
        return 100.0 - (100.0 / (1.0 + rs));

    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] in update_rsi: " << e.what() << std::endl;
        return 50.0; // ���ذ�ȫֵ
    }


}
