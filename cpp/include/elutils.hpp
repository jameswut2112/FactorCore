//#pragma once
//#include <Eigen/Dense>
//#include <vector>
//#include <cmath>
//#include <algorithm>
//#include <tuple>
//
//namespace elutils {
//
//    // �Ƴ���ֵ������������滻�������
//    std::tuple<Eigen::VectorXi, Eigen::VectorXd> remove_zero(const Eigen::VectorXd& data) {
//        Eigen::VectorXi mask = (data.array() == 0).cast<int>();
//        Eigen::VectorXd result = data;
//        for (int i = 0; i < result.size(); i++) {
//            if (mask(i)) result(i) = 1.0;
//        }
//        return { mask, result };
//    }
//
//    // ����ѡ���� (��Ԫ��������������汾)
//    Eigen::VectorXd iif(const Eigen::VectorXi& condition,
//        const Eigen::VectorXd& true_val,
//        const Eigen::VectorXd& false_val) {
//        return condition.cast<double>().cwiseProduct(true_val) +
//            (1 - condition.cast<double>()).cwiseProduct(false_val);
//    }
//
//    // �̶����ڵ����ֵ (HHV)
//    Eigen::VectorXd HHV(const Eigen::VectorXd& data, int period) {
//        if (period <= 1) return data;
//
//        Eigen::VectorXd result(data.size());
//        for (int i = 0; i < data.size(); i++) {
//            int start = std::max(0, i - period + 1);
//            result(i) = data.segment(start, i - start + 1).maxCoeff();
//        }
//        return result;
//    }
//
//    // �ɱ䴰�ڵ����ֵ (HHVm)
//    Eigen::VectorXd HHVm(const Eigen::VectorXd& data, const Eigen::VectorXi& periods) {
//        Eigen::VectorXd result(data.size());
//        for (int i = 0; i < data.size(); i++) {
//            int window = periods(i);
//            if (window <= 0) {
//                result(i) = (i > 0) ? std::max(data(i), result(i - 1)) : data(i);
//            }
//            else {
//                int start = std::max(0, i - window + 1);
//                result(i) = data.segment(start, i - start + 1).maxCoeff();
//            }
//        }
//        return result;
//    }
//
//    // �̶����ڵ����ֵ (LLV)
//    Eigen::VectorXd LLV(const Eigen::VectorXd& data, int period) {
//        if (period <= 1) return data;
//
//        Eigen::VectorXd result(data.size());
//        for (int i = 0; i < data.size(); i++) {
//            int start = std::max(0, i - period + 1);
//            result(i) = data.segment(start, i - start + 1).minCoeff();
//        }
//        return result;
//    }
//
//    // �ɱ䴰�ڵ����ֵ (LLVm)
//    Eigen::VectorXd LLVm(const Eigen::VectorXd& data, const Eigen::VectorXi& periods) {
//        Eigen::VectorXd result(data.size());
//        for (int i = 0; i < data.size(); i++) {
//            int window = periods(i);
//            if (window <= 0) {
//                result(i) = (i > 0) ? std::min(data(i), result(i - 1)) : data(i);
//            }
//            else {
//                int start = std::max(0, i - window + 1);
//                result(i) = data.segment(start, i - start + 1).minCoeff();
//            }
//        }
//        return result;
//    }
//
//    // �ƶ�ƽ�� (MA)
//    Eigen::VectorXd MA(const Eigen::VectorXd& data, int period) {
//        if (period <= 1) return data;
//
//        Eigen::VectorXd result(data.size());
//        double sum = 0.0;
//        int count = 0;
//
//        for (int i = 0; i < data.size(); i++) {
//            sum += data(i);
//            count++;
//
//            if (count > period) {
//                sum -= data(i - period);
//                count = period;
//            }
//
//            result(i) = sum / count;
//        }
//        return result;
//    }
//
//    // ָ���ƶ�ƽ�� (EMA)
//    Eigen::VectorXd EMA(const Eigen::VectorXd& data, int period) {
//        if (period <= 1) return data;
//
//        Eigen::VectorXd result(data.size());
//        double alpha = 2.0 / (period + 1);
//        result(0) = data(0);
//
//        for (int i = 1; i < data.size(); i++) {
//            result(i) = alpha * data(i) + (1 - alpha) * result(i - 1);
//        }
//        return result;
//    }
//
//    // ������� (SUM)
//    Eigen::VectorXd SUM(const Eigen::VectorXd& data, int period) {
//        if (period <= 1) return data;
//
//        Eigen::VectorXd result(data.size());
//        double sum = 0.0;
//        int count = 0;
//
//        for (int i = 0; i < data.size(); i++) {
//            sum += data(i);
//            count++;
//
//            if (count > period) {
//                sum -= data(i - period);
//                count = period;
//            }
//
//            result(i) = sum;
//        }
//        return result;
//    }
//
//    // ���ǿ��ָ�� (RSI)
//    Eigen::VectorXd RSI(const Eigen::VectorXd& prices, int period) {
//        Eigen::VectorXd gains = Eigen::VectorXd::Zero(prices.size());
//        Eigen::VectorXd losses = Eigen::VectorXd::Zero(prices.size());
//
//        // ����۸�仯
//        for (int i = 1; i < prices.size(); i++) {
//            double diff = prices(i) - prices(i - 1);
//            gains(i) = std::max(diff, 0.0);
//            losses(i) = std::max(-diff, 0.0);
//        }
//
//        // ����ƽ�������ƽ����ʧ
//        Eigen::VectorXd avg_gain = EMA(gains, period);
//        Eigen::VectorXd avg_loss = EMA(losses, period);
//
//        // ����RSI
//        Eigen::VectorXd rs = avg_gain.array() / (avg_loss.array() + 1e-9);
//        return 100.0 - (100.0 / (1.0 + rs.array()));
//    }
//
//    // ���ִ� (Bollinger Bands)
//    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
//        BBANDS(const Eigen::VectorXd& data, int period, double sigma) {
//        Eigen::VectorXd middle = MA(data, period);
//        Eigen::VectorXd std_dev = Eigen::VectorXd::Zero(data.size());
//
//        // �����׼��
//        for (int i = period - 1; i < data.size(); i++) {
//            int start = i - period + 1;
//            Eigen::VectorXd segment = data.segment(start, period);
//            double mean = segment.mean();
//            double sq_sum = (segment.array() - mean).square().sum();
//            std_dev(i) = std::sqrt(sq_sum / period);
//        }
//
//        Eigen::VectorXd upper = middle.array() + sigma * std_dev.array();
//        Eigen::VectorXd lower = middle.array() - sigma * std_dev.array();
//
//        return { upper, middle, lower };
//    }
//
//    // MACDָ��
//    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
//        MACD(const Eigen::VectorXd& prices, int fast_period, int slow_period, int signal_period) {
//        Eigen::VectorXd fast_ema = EMA(prices, fast_period);
//        Eigen::VectorXd slow_ema = EMA(prices, slow_period);
//
//        Eigen::VectorXd macd_line = fast_ema - slow_ema;
//        Eigen::VectorXd signal_line = EMA(macd_line, signal_period);
//        Eigen::VectorXd histogram = macd_line - signal_line;
//
//        return { macd_line, signal_line, histogram };
//    }
//
//    // ����ǰN��ֵ (REF)
//    Eigen::VectorXd REF(const Eigen::VectorXd& data, int n) {
//        if (n == 0) return data;
//
//        Eigen::VectorXd result(data.size());
//        if (n > 0) {
//            result.head(n) = data.head(n);
//            result.tail(data.size() - n) = data.head(data.size() - n);
//        }
//        else {
//            int abs_n = std::abs(n);
//            result.head(data.size() - abs_n) = data.tail(data.size() - abs_n);
//            result.tail(abs_n) = data.tail(abs_n);
//        }
//        return result;
//    }
//
//    // �����ź� (CROSS)
//    Eigen::VectorXi CROSS(const Eigen::VectorXd& series1, const Eigen::VectorXd& series2) {
//        Eigen::VectorXi result = Eigen::VectorXi::Zero(series1.size());
//        for (int i = 1; i < series1.size(); i++) {
//            if (series1(i - 1) < series2(i - 1) && series1(i) > series2(i)) {
//                result(i) = 1;
//            }
//        }
//        return result;
//    }
//
//    // �۸�䶯�� (ROC)
//    Eigen::VectorXd ROC(const Eigen::VectorXd& data, int period, bool absmode = false) {
//        Eigen::VectorXd ref_data = REF(data, period);
//        if (absmode) {
//            return ((data.array() - ref_data.array()) / ref_data.array().abs()) * 100.0;
//        }
//        return ((data.array() - ref_data.array()) / ref_data.array()) * 100.0;
//    }
//
//    // �ٷֱ����� (Percent Rank)
//    Eigen::VectorXd PERCENT_RANK(const Eigen::VectorXd& data, int period) {
//        Eigen::VectorXd result = Eigen::VectorXd::Zero(data.size());
//        for (int i = period; i < data.size(); i++) {
//            int start = i - period + 1;
//            Eigen::VectorXd segment = data.segment(start, period);
//            double current = data(i);
//            double count = (segment.array() <= current).count();
//            result(i) = (count / period) * 100.0;
//        }
//        return result;
//    }
//
//    // ����ֵ���� (VALUE_WHEN)
//    Eigen::VectorXd VALUE_WHEN(const Eigen::VectorXi& condition, const Eigen::VectorXd& data) {
//        Eigen::VectorXd result(data.size());
//        double last_value = 0.0;
//        for (int i = 0; i < data.size(); i++) {
//            if (condition(i)) {
//                last_value = data(i);
//            }
//            result(i) = last_value;
//        }
//        return result;
//    }
//
//    // ���ֵ���� (HIGHEST_SINCE)
//    Eigen::VectorXd HIGHEST_SINCE(const Eigen::VectorXi& condition, const Eigen::VectorXd& data) {
//        Eigen::VectorXd result(data.size());
//        double current_high = data(0);
//        bool active = false;
//
//        for (int i = 0; i < data.size(); i++) {
//            if (condition(i)) {
//                current_high = data(i);
//                active = true;
//            }
//            else if (active) {
//                current_high = std::max(current_high, data(i));
//            }
//            result(i) = current_high;
//        }
//        return result;
//    }
//
//    // ���ֵ���� (LOWEST_SINCE)
//    Eigen::VectorXd LOWEST_SINCE(const Eigen::VectorXi& condition, const Eigen::VectorXd& data) {
//        Eigen::VectorXd result(data.size());
//        double current_low = data(0);
//        bool active = false;
//
//        for (int i = 0; i < data.size(); i++) {
//            if (condition(i)) {
//                current_low = data(i);
//                active = true;
//            }
//            else if (active) {
//                current_low = std::min(current_low, data(i));
//            }
//            result(i) = current_low;
//        }
//        return result;
//    }
//
//    // ��׼�� (STDEV)
//    Eigen::VectorXd STDEV(const Eigen::VectorXd& data, int period) {
//        Eigen::VectorXd result(data.size());
//        for (int i = period - 1; i < data.size(); i++) {
//            int start = i - period + 1;
//            Eigen::VectorXd segment = data.segment(start, period);
//            double mean = segment.mean();
//            double sq_sum = (segment.array() - mean).square().sum();
//            result(i) = std::sqrt(sq_sum / period);
//        }
//        return result;
//    }
//
//    // ����ķ˹ %R (WILLR)
//    Eigen::VectorXd WILLR(const Eigen::VectorXd& high,
//        const Eigen::VectorXd& low,
//        const Eigen::VectorXd& close,
//        int period) {
//        Eigen::VectorXd result(close.size());
//        for (int i = period - 1; i < close.size(); i++) {
//            int start = i - period + 1;
//            double highest = high.segment(start, period).maxCoeff();
//            double lowest = low.segment(start, period).minCoeff();
//            result(i) = (highest - close(i)) / (highest - lowest + 1e-9) * -100.0;
//        }
//        return result;
//    }
//
//    // ��Ʒͨ��ָ�� (CCI)
//    Eigen::VectorXd CCI(const Eigen::VectorXd& high,
//        const Eigen::VectorXd& low,
//        const Eigen::VectorXd& close,
//        int period) {
//        Eigen::VectorXd tp = (high + low + close) / 3.0;
//        Eigen::VectorXd sma = MA(tp, period);
//        Eigen::VectorXd mean_dev = Eigen::VectorXd::Zero(close.size());
//
//        for (int i = period - 1; i < close.size(); i++) {
//            int start = i - period + 1;
//            Eigen::VectorXd segment = tp.segment(start, period);
//            double mean = sma(i);
//            double dev_sum = (segment.array() - mean).abs().sum();
//            mean_dev(i) = dev_sum / period;
//        }
//
//        return (tp - sma).array() / (0.015 * mean_dev.array() + 1e-9);
//    }
//
//} // namespace elutils





#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <stdexcept>

namespace elutils {

    // �޸��� remove_zero ����
    std::tuple<Eigen::VectorXi, Eigen::VectorXd> remove_zero(const Eigen::VectorXd& data) {
        Eigen::VectorXi mask = (data.array() == 0.0).cast<int>();
        Eigen::VectorXd result = data;

        // ʹ��Eigen���������������ѭ��
        Eigen::ArrayXd::Index size = result.size();
        for (Eigen::ArrayXd::Index i = 0; i < size; ++i) {
            if (mask(i)) result(i) = 1.0;
        }
        return { mask, result };
    }

    // �Ż��� iif ����
    Eigen::VectorXd iif(const Eigen::VectorXi& condition,
        const Eigen::VectorXd& true_val,
        const Eigen::VectorXd& false_val) {
        return condition.cast<double>().asDiagonal() * true_val +
            (Eigen::VectorXi::Ones(condition.size()) - condition).cast<double>().asDiagonal() * false_val;
    }

    // �Ż��� HHV ����
    Eigen::VectorXd HHV(const Eigen::VectorXd& data, int period) {
        if (period <= 1) return data;

        int size = data.size();
        Eigen::VectorXd result(size);

        // ����ǰperiod-1��Ԫ��
        result(0) = data(0);
        for (int i = 1; i < std::min(period, size); i++) {
            result(i) = std::max(data(i), result(i - 1));
        }

        // ʹ�û������ڼ���ʣ��Ԫ��
        for (int i = period; i < size; i++) {
            result(i) = data.segment(i - period + 1, period).maxCoeff();
        }
        return result;
    }

    // �Ż��� LLV ����
    Eigen::VectorXd LLV(const Eigen::VectorXd& data, int period) {
        if (period <= 1) return data;

        int size = data.size();
        Eigen::VectorXd result(size);

        // ����ǰperiod-1��Ԫ��
        result(0) = data(0);
        for (int i = 1; i < std::min(period, size); i++) {
            result(i) = std::min(data(i), result(i - 1));
        }

        // ʹ�û������ڼ���ʣ��Ԫ��
        for (int i = period; i < size; i++) {
            result(i) = data.segment(i - period + 1, period).minCoeff();
        }
        return result;
    }

    // �Ż����ƶ�ƽ�� (MA)
    Eigen::VectorXd MA(const Eigen::VectorXd& data, int period) {
        if (period <= 1) return data;

        int size = data.size();
        Eigen::VectorXd result(size);
        double sum = 0.0;
        int count = 0;

        for (int i = 0; i < size; i++) {
            sum += data(i);
            count++;

            if (count > period) {
                sum -= data(i - period);
                count = period;
            }

            result(i) = sum / count;
        }
        return result;
    }

    // �Ż���ָ���ƶ�ƽ�� (EMA)
    Eigen::VectorXd EMA(const Eigen::VectorXd& data, int period) {
        if (period <= 1) return data;

        int size = data.size();
        Eigen::VectorXd result(size);
        double alpha = 2.0 / (period + 1.0);
        result(0) = data(0);

        for (int i = 1; i < size; i++) {
            result(i) = alpha * data(i) + (1 - alpha) * result(i - 1);
        }
        return result;
    }

    // �Ż��Ĺ������ (SUM)
    Eigen::VectorXd SUM(const Eigen::VectorXd& data, int period) {
        if (period <= 1) return data;

        int size = data.size();
        Eigen::VectorXd result(size);
        double sum = 0.0;
        int count = 0;

        for (int i = 0; i < size; i++) {
            sum += data(i);
            count++;

            if (count > period) {
                sum -= data(i - period);
                count = period;
            }

            result(i) = sum;
        }
        return result;
    }

    // �Ż��� RSI ����
    Eigen::VectorXd RSI(const Eigen::VectorXd& prices, int period) {
        int size = prices.size();
        if (size < period + 1) {
            return Eigen::VectorXd::Constant(size, std::numeric_limits<double>::quiet_NaN());
        }

        Eigen::VectorXd gains = Eigen::VectorXd::Zero(size);
        Eigen::VectorXd losses = Eigen::VectorXd::Zero(size);

        // ����۸�仯
        for (int i = 1; i < size; i++) {
            double diff = prices(i) - prices(i - 1);
            gains(i) = std::max(diff, 0.0);
            losses(i) = std::max(-diff, 0.0);
        }

        // ����ƽ�������ƽ����ʧ
        Eigen::VectorXd avg_gain = EMA(gains, period);
        Eigen::VectorXd avg_loss = EMA(losses, period);

        // ����RSI
        Eigen::VectorXd rs = avg_gain.array() / (avg_loss.array() + 1e-9);
        return 100.0 - (100.0 / (1.0 + rs.array()));
    }

    // �Ż��� REF ����
    Eigen::VectorXd REF(const Eigen::VectorXd& data, int n) {
        if (n == 0) return data;

        int size = data.size();
        Eigen::VectorXd result(size);

        if (n > 0) {
            if (n >= size) {
                result.setConstant(data(0));
            }
            else {
                result.head(n) = data.head(n);
                result.tail(size - n) = data.head(size - n);
            }
        }
        else {
            int abs_n = std::abs(n);
            if (abs_n >= size) {
                result.setConstant(data(size - 1));
            }
            else {
                result.head(size - abs_n) = data.tail(size - abs_n);
                result.tail(abs_n) = data.tail(abs_n);
            }
        }
        return result;
    }

    // �����������������Ż�...

} // namespace elutils
