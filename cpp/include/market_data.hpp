#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

// �г���������ö��
enum class MarketDataType {
    OPEN,   // ���̼�
    HIGH,   // ��߼�
    LOW,    // ��ͼ�
    CLOSE,  // ���̼�
    VOLUME  // �ɽ���
};

// �г����ݽṹ
struct MarketData {
    Eigen::VectorXd open;
    Eigen::VectorXd high;
    Eigen::VectorXd low;
    Eigen::VectorXd close;
    Eigen::VectorXd volume;

    // ͳһ���ʽӿ�
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

    // ������֤
    void validate() const {
        const size_t size = close.size();
        if (open.size() != size) throw std::runtime_error("Open price size mismatch");
        if (high.size() != size) throw std::runtime_error("High price size mismatch");
        if (low.size() != size) throw std::runtime_error("Low price size mismatch");
        if (volume.size() != size) throw std::runtime_error("Volume size mismatch");
    }

    // ��ȡ���ݳ���
    size_t size() const { return close.size(); }

    // ��ȡ�������ݵ�
    double latest_open() const { return open.size() > 0 ? open[open.size() - 1] : 0.0; }
    double latest_high() const { return high.size() > 0 ? high[high.size() - 1] : 0.0; }
    double latest_low() const { return low.size() > 0 ? low[low.size() - 1] : 0.0; }
    double latest_close() const { return close.size() > 0 ? close[close.size() - 1] : 0.0; }
    double latest_volume() const { return volume.size() > 0 ? volume[volume.size() - 1] : 0.0; }

    // ��ȡ��Ƭ����
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
