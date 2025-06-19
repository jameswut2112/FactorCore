#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

// 市场数据类型枚举
enum class MarketDataType {
    OPEN,   // 开盘价
    HIGH,   // 最高价
    LOW,    // 最低价
    CLOSE,  // 收盘价
    VOLUME  // 成交量
};

// 市场数据结构
struct MarketData {
    Eigen::VectorXd open;
    Eigen::VectorXd high;
    Eigen::VectorXd low;
    Eigen::VectorXd close;
    Eigen::VectorXd volume;

    // 统一访问接口
    const Eigen::VectorXd& get(MarketDataType type) const {
        switch (type) {
        case MarketDataType::OPEN: return open;
        case MarketDataType::HIGH: return high;
        case MarketDataType::LOW: return low;
        case MarketDataType::CLOSE: return close;
        case MarketDataType::VOLUME: return volume;
        default: throw std::invalid_argument("Invalid market data type");
        }
    }

    // 数据验证
    void validate() const {
        const size_t size = close.size();
        if (open.size() != size) throw std::runtime_error("Open price size mismatch");
        if (high.size() != size) throw std::runtime_error("High price size mismatch");
        if (low.size() != size) throw std::runtime_error("Low price size mismatch");
        if (volume.size() != size) throw std::runtime_error("Volume size mismatch");
    }

    // 获取数据长度
    size_t size() const { return close.size(); }

    // 获取最新数据点
    double latest_open() const { return open.size() > 0 ? open[open.size() - 1] : 0.0; }
    double latest_high() const { return high.size() > 0 ? high[high.size() - 1] : 0.0; }
    double latest_low() const { return low.size() > 0 ? low[low.size() - 1] : 0.0; }
    double latest_close() const { return close.size() > 0 ? close[close.size() - 1] : 0.0; }
    double latest_volume() const { return volume.size() > 0 ? volume[volume.size() - 1] : 0.0; }

    // 获取切片数据
    MarketData slice(size_t start, size_t end) const {
        MarketData sliced;
        sliced.open = open.segment(start, end - start);
        sliced.high = high.segment(start, end - start);
        sliced.low = low.segment(start, end - start);
        sliced.close = close.segment(start, end - start);
        sliced.volume = volume.segment(start, end - start);
        return sliced;
    }
};
