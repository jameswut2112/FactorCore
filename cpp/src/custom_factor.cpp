//#include "custom_factor.hpp"
//#include <cmath>
//
//Eigen::VectorXd CustomFactor::calculate(
//    const Eigen::MatrixXd& data,
//    const Eigen::VectorXd& params)
//{
//    // 示例计算：加权移动平均
//    const int n = data.rows();
//    const int window = static_cast<int>(params[0]);
//
//    if (window <= 0 || window > n) {
//        throw std::invalid_argument("无效的窗口大小");
//    }
//
//    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
//
//    for (int i = window - 1; i < n; ++i) {
//        double sum = 0.0;
//        for (int j = 0; j < window; ++j) {
//            sum += data(i - j, 0); // 假设使用第一列
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
    // 设置默认参数范围
    setParaRange(
        { 1, 1 },     // pmin
        { 200, 100 }, // pmax
        { 1, 1 },     // pstep
        { 15, 60 }    // default_params
    );

    // 初始化状态变量
    prev_avg_gain = 0.0;
    prev_avg_loss = 0.0;
}

// 全量计算因子值
void TR0::eval_full(const MarketData& market_data,
    const std::vector<int>& params) {
    /*const int period = std::max(2, params[0]);
    const int n = market_data.close.size();*/

    if (params.empty() || params[0] <= 1) {
        throw std::invalid_argument("Invalid RSI period parameter");
    }

    const int period = params[0];
    const int n = market_data.close.size();

	// 数据长度验证，确保有足够的数据点进行RSI计算
    if (n < period + 1) {
        value.resize(n);
        value.setConstant(std::numeric_limits<double>::quiet_NaN());
        std::cerr << "[WARNING] Insufficient data for RSI calculation. Need "
            << period + 1 << " points, got " << n << std::endl;
        return;
    }


    // 安全计算delta_size
    const int delta_size = n - 1;
    if (delta_size <= 0 || delta_size >= market_data.close.size()) {
        throw std::runtime_error("Invalid delta calculation");
    }


    try {
        // 使用segment替代head/tail更安全
        const Eigen::VectorXd deltas = market_data.close.segment(1, delta_size) -
            market_data.close.segment(0, delta_size);

        // 计算增益和损失
        const Eigen::VectorXd gains = deltas.cwiseMax(0.0);
        const Eigen::VectorXd losses = (-deltas).cwiseMax(0.0);

        // 初始化RSI值
        value.resize(n);
        value.setZero();

        // 计算初始平均增益和损失（前period个点）
        double avg_gain = gains.head(period).mean();
        double avg_loss = losses.head(period).mean();

        // 计算前period+1点的RSI（第period点）
        if (avg_loss < 1e-9) {
            value(period) = (avg_gain < 1e-9) ? 50.0 : 100.0;
        }
        else {
            const double rs = avg_gain / avg_loss;
            value(period) = 100.0 - (100.0 / (1.0 + rs));
        }

        // 计算后续点的RSI
        for (int i = period + 1; i < n; ++i) {
            // 更新平均增益和损失（使用Wilder平滑）
            avg_gain = (avg_gain * (period - 1) + gains(i - 1)) / period;
            avg_loss = (avg_loss * (period - 1) + losses(i - 1)) / period;

            // 计算RS
            const double rs = (avg_loss < 1e-9) ?
                std::numeric_limits<double>::infinity() :
                avg_gain / avg_loss;

            // 计算RSI
            value(i) = 100.0 - (100.0 / (1.0 + rs));
        }

        // ... 其余计算逻辑保持不变 ...
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] in eval_full: " << e.what() << std::endl;
        throw;
    }


    // 保存状态用于增量计算
    save_rsi_state(market_data, period);
}

// 增量更新因子值
void TR0::step_update(const MarketData& market_data,
    const std::vector<int>& params) {
    const int new_size = market_data.size();
    const int current_size = value.size();

    // 严格验证数据长度差
    if (new_size <= current_size) {
        throw std::runtime_error("New data size must be greater than current size");
    }

    if (new_size - current_size > 1) {
        // 如果数据差距太大，回退到全量计算
        eval_full(market_data, params);
        return;
    }

    // 执行增量更新
    const double new_rsi = update_rsi(market_data, params[0]);

    // 扩展value向量
    Eigen::VectorXd new_value(current_size + 1);
    new_value << value, new_rsi;
    value = new_value;
}

// 生成交易信号
std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
TR0::value_to_signal(const MarketData& market_data,
    const std::vector<int>& params) {
    // 如果值未计算，先进行全量计算
    if (value.size() == 0) {
        eval_full(market_data, params);
    }

    const double threshold = 50.0 + 0.5 * params[1];
    const int n = value.size();

    // 使用Eigen的向量化操作生成信号
    Eigen::VectorXi buy_signal = (value.array() > threshold).cast<int>();
    Eigen::VectorXi sell_signal = (value.array() < (100.0 - threshold)).cast<int>();
    Eigen::VectorXi close_signal = Eigen::VectorXi::Zero(n);

    return std::make_tuple(buy_signal, sell_signal, close_signal);
}

//// 保存RSI计算状态
//void TR0::save_rsi_state(const MarketData& market_data, int period) {
//    const int n = market_data.size();
//
//    if (n < period + 1) {
//        // 数据不足，保存全部
//        price_window = market_data;
//        return;
//    }
//
//    // 保存最后period+1个价格点
//    price_window = market_data.slice(n - period - 1, n);
//}

void TR0::save_rsi_state(const MarketData& market_data, int period) {
    const int n = market_data.close.size();

    if (n == 0) {
        price_window = MarketData(); // 重置为空
        return;
    }

    // 计算安全范围
    const int start_idx = std::max(0, n - period - 1);
    const int length = n - start_idx;

    // 使用安全切片方法
    price_window = market_data.slice(start_idx, n);
}


// 更新RSI值（增量）
double TR0::update_rsi(const MarketData& market_data, int period) {

    // 参数验证
    if (period <= 1) {
        throw std::invalid_argument("RSI period must be greater than 1");
    }

    // Replace the usage of `back()` with the appropriate Eigen method to access the last element.  
    // 安全获取最后一个收盘价
    const double new_close = market_data.close[market_data.close.size() - 1];
    //const double new_close = market_data.close.size() > 0 ? market_data.close(market_data.close.size() - 1) : 0.0;

    // 更新价格窗口
    if (price_window.size() == 0) {
        // 首次更新，初始化窗口
        price_window = market_data;
        return 50.0; // 默认RSI值
    }

    try {
        // 创建新窗口副本
        MarketData new_window = price_window;

        // 添加新数据点
        new_window.close.conservativeResize(new_window.close.size() + 1);
        new_window.close.tail(1) << new_close;

        // 修剪窗口大小
        if (new_window.close.size() > period + 1) {
            new_window.close = new_window.close.tail(period + 1);
        }

        price_window = new_window;

        // 检查是否有足够数据计算
        if (price_window.close.size() < 2) {
            return 50.0;
        }

        // 计算最新delta
        const double delta = price_window.close[price_window.close.size() - 1] -
            price_window.close[price_window.close.size() - 2];

        const double gain = std::max(delta, 0.0);
        const double loss = std::max(-delta, 0.0);

        // 更新平均增益和损失
        prev_avg_gain = (prev_avg_gain * (period - 1) + gain) / period;
        prev_avg_loss = (prev_avg_loss * (period - 1) + loss) / period;

        // 处理除零情况
        if (prev_avg_loss < 1e-9) {
            return (prev_avg_gain < 1e-9) ? 50.0 : 100.0;
        }

        // 计算RS
        const double rs = prev_avg_gain / prev_avg_loss;

        // 返回最新RSI值
        return 100.0 - (100.0 / (1.0 + rs));

    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] in update_rsi: " << e.what() << std::endl;
        return 50.0; // 返回安全值
    }


}
