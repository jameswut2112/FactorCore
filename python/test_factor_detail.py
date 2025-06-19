import factors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StructuredData:
    def __init__(self, close=None, **kwargs):
        """
        简化版数据结构（兼容旧测试）
        用法：
        - 旧方式：StructuredData(close=prices)
        - 新方式：StructuredData(open=..., high=..., low=..., close=..., vol=...)
        """
        # 必须字段（旧测试只需要close）
        self.close = np.array(close) if close is not None else np.array([])
        
        # 新字段带默认值（兼容旧测试）
        self.open = kwargs.get('open', self.close.copy())          # 默认同close
        self.high = kwargs.get('high', self.close.copy())          # 默认同close
        self.low = kwargs.get('low', self.close.copy())            # 默认同close
        self.vol = kwargs.get('vol', np.ones_like(self.close))     # 默认全1
        self.open_int = kwargs.get('open_int', np.zeros_like(self.close))  # 默认全0

    @property
    def length(self):
        """数据长度（兼容旧属性）"""
        return len(self.close)

    def __repr__(self):
        return f"<StructuredData len={self.length}>"

def sample_data():
    """生成符合实际交易场景的模拟数据（含涨跌停限制）"""
    np.random.seed(42)
    size = 8000  # 交易日数量

    # 基础价格序列（初始价设为100）
    price_base = np.zeros(size)
    price_base[0] = 100

    # 生成每日涨跌幅（考虑涨跌停限制）
    for i in range(1, size):
        # 生成-10%到+10%之间的涨跌幅，符合正态分布特性
        # 大部分涨跌幅集中在0附近，极端值概率较低
        daily_change = np.random.normal(0, 0.02)  # 标准差设为0.02，控制波动范围

        # 应用涨跌停限制
        daily_change = max(-0.1, min(0.1, daily_change))

        # 计算当日价格
        price_base[i] = price_base[i - 1] * (1 + daily_change)

    # 生成开盘价（考虑与前一日收盘价的关系）
    open_price = np.zeros(size)
    open_price[0] = price_base[0]

    for i in range(1, size):
        # 开盘价通常接近前一日收盘价，但有小幅波动
        open_change = np.random.normal(0, 0.005)  # 开盘波动通常较小
        open_price[i] = price_base[i - 1] * (1 + open_change)

    # 生成最高价和最低价（确保逻辑合理性）
    high_price = np.zeros(size)
    low_price = np.zeros(size)

    for i in range(size):
        # 生成波动因子（确保最高价≥收盘价≥开盘价≥最低价）
        high_factor = 1 + np.random.uniform(0, 0.015)  # 最高价通常比收盘价高0-1.5%
        low_factor = 1 - np.random.uniform(0, 0.015)  # 最低价通常比收盘价低0-1.5%

        # 确保最高价不超过涨停限制
        high_limit = price_base[i - 1] * 1.1 if i > 0 else price_base[i] * 1.1
        high_price[i] = min(price_base[i] * high_factor, high_limit)

        # 确保最低价不低于跌停限制
        low_limit = price_base[i - 1] * 0.9 if i > 0 else price_base[i] * 0.9
        low_price[i] = max(price_base[i] * low_factor, low_limit)

        # 处理可能的边界情况（确保high≥close≥low且high≥open≥low）
        high_price[i] = max(high_price[i], open_price[i], price_base[i])
        low_price[i] = min(low_price[i], open_price[i], price_base[i])

    # 生成成交量（与价格波动相关）
    volume = np.zeros(size)
    for i in range(size):
        # 计算价格波动幅度
        price_change = abs(price_base[i] / price_base[i - 1] - 1) if i > 0 else 0

        # 波动越大，成交量越高（模拟市场情绪）
        base_vol = np.random.randint(10000, 100000)  # 基础成交量
        vol_factor = 1 + price_change * 5  # 波动放大因子
        volume[i] = int(base_vol * vol_factor)

    # 加入一些异常交易日（如涨停/跌停日成交量异常）
    for i in range(size):
        if i > 0:
            # 涨停日
            if np.isclose(price_base[i], price_base[i - 1] * 1.1, atol=1e-4):
                volume[i] = int(volume[i] * 1.5)  # 涨停日成交量通常放大
            # 跌停日
            elif np.isclose(price_base[i], price_base[i - 1] * 0.9, atol=1e-4):
                volume[i] = int(volume[i] * 1.2)  # 跌停日成交量可能略有放大

    return StructuredData(
        close=price_base,
        open=open_price,
        high=high_price,
        low=low_price,
        vol=volume
    )

def structured_to_market_data(structured_data):
    """将StructuredData转换为MarketData"""
    market_data = factors.MarketData()
    market_data.open = structured_data.open.astype(np.float64)
    market_data.high = structured_data.high.astype(np.float64)
    market_data.low = structured_data.low.astype(np.float64)
    market_data.close = structured_data.close.astype(np.float64)
    market_data.volume = structured_data.vol.astype(np.float64)
    
    # 验证数据一致性
    try:
        market_data.validate()
        print("Market data validated successfully.")
    except Exception as e:
        print(f"Market data validation failed: {str(e)}")
        return None
    
    return market_data

def test_factor_performance():
    """测试因子计算性能"""
    print("\n===== Testing Factor Performance =====")
    
    # 生成模拟数据
    print("Generating simulated market data...")
    structured_data = sample_data()
    market_data = structured_to_market_data(structured_data)
    
    if market_data is None:
        return
    
    print(f"Generated {market_data.size()} data points.")
    
    # 创建因子
    rsi_factor = factors.TR0()
    
    # 测试全量计算性能
    print("\nTesting full calculation performance...")
    start_time = datetime.now()
    buy, sell, close = rsi_factor.eval(market_data)
    full_duration = datetime.now() - start_time
    print(f"Full calculation completed in {full_duration.total_seconds():.4f} seconds")
    
    # 测试增量更新性能
    print("\nTesting incremental update performance...")
    
    # 创建新数据（添加一个点）
    new_structured_data = StructuredData(
        close=np.append(structured_data.close, 105.0),
        open=np.append(structured_data.open, 104.5),
        high=np.append(structured_data.high, 106.0),
        low=np.append(structured_data.low, 104.0),
        vol=np.append(structured_data.vol, 20000)
    )
    new_market_data = structured_to_market_data(new_structured_data)
    
    start_time = datetime.now()
    buy, sell, close = rsi_factor.eval(new_market_data)
    incremental_duration = datetime.now() - start_time
    print(f"Incremental update completed in {incremental_duration.total_seconds():.4f} seconds")
    
    # 性能比较
    print(f"\nPerformance improvement: {full_duration/incremental_duration:.1f}x faster")
    
    # 获取因子值
    rsi_values = rsi_factor.get_value()
    
    # 绘制最后200个点的结果
    plot_results(market_data.close[-200:], rsi_values[-200:], 
                 buy[-200:], sell[-200:], "Performance Test")

def test_large_scale():
    """测试大规模数据处理能力"""
    print("\n===== Testing Large Scale Data Handling =====")
    
    # 生成大规模模拟数据
    print("Generating large-scale market data (20,000 points)...")
    np.random.seed(42)
    large_size = 20000
    
    # 直接生成大规模数据（避免使用sample_data函数以节省时间）
    prices = np.cumprod(1 + np.random.normal(0, 0.01, large_size)) * 100
    volumes = np.random.randint(10000, 200000, large_size)
    
    structured_data = StructuredData(
        close=prices,
        open=prices * (1 + np.random.normal(0, 0.002, large_size)),
        high=prices * (1 + np.random.uniform(0, 0.015, large_size)),
        low=prices * (1 - np.random.uniform(0, 0.015, large_size)),
        vol=volumes
    )
    
    market_data = structured_to_market_data(structured_data)
    
    if market_data is None:
        return
    
    # 创建因子
    rsi_factor = factors.TR0()
    
    # 测试计算性能
    print("\nTesting calculation performance on large dataset...")
    start_time = datetime.now()
    buy, sell, close = rsi_factor.eval(market_data)
    duration = datetime.now() - start_time
    print(f"Calculated RSI for {market_data.size()} data points in {duration.total_seconds():.4f} seconds")
    
    # 计算每秒处理的数据点
    points_per_second = market_data.size() / duration.total_seconds()
    print(f"Processing speed: {points_per_second:.2f} data points per second")
    
    # 绘制部分结果
    plot_results(market_data.close[-500:], rsi_factor.get_value()[-500:], 
                 buy[-500:], sell[-500:], "Large Scale Test")

def test_trading_strategy():
    """测试交易策略表现"""
    print("\n===== Testing Trading Strategy =====")
    
    # 生成模拟数据
    print("Generating simulated market data...")
    structured_data = sample_data()
    market_data = structured_to_market_data(structured_data)
    
    if market_data is None:
        return
    
    # 创建因子
    rsi_factor = factors.TR0()
    
    # 计算交易信号
    buy, sell, close = rsi_factor.eval(market_data, [14, 30])
    rsi_values = rsi_factor.get_value()
    
    # 模拟交易
    cash = 1000000  # 初始资金
    position = 0    # 持仓数量
    trade_history = []
    portfolio_values = []
    
    for i in range(len(market_data.close)):
        price = market_data.close[i]
        
        # 买入信号
        if buy[i] and cash > price * 100:
            shares = min(100, int(cash / price))
            cost = shares * price
            cash -= cost
            position += shares
            trade_history.append(('BUY', i, price, shares))
        
        # 卖出信号
        elif sell[i] and position > 0:
            shares = min(100, position)
            revenue = shares * price
            cash += revenue
            position -= shares
            trade_history.append(('SELL', i, price, shares))
        
        # 计算当前资产价值
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)
    
    # 计算策略表现
    initial_value = 1000000
    final_value = portfolio_values[-1]
    returns = (final_value - initial_value) / initial_value * 100
    
    # 计算基准表现
    buy_hold_value = 1000000 / market_data.close[0] * market_data.close[-1]
    buy_hold_returns = (buy_hold_value - initial_value) / initial_value * 100
    
    print("\n===== Trading Results =====")
    print(f"Initial capital: ${initial_value:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Strategy returns: {returns:.2f}%")
    print(f"Buy & hold returns: {buy_hold_returns:.2f}%")
    print(f"Number of trades: {len(trade_history)}")
    
    # 绘制结果
    plot_trading_results(market_data.close, rsi_values, buy, sell, 
                         portfolio_values, "Trading Strategy Test")

def plot_results(prices, rsi, buy_signals, sell_signals, title):
    """绘制价格、RSI和交易信号"""
    plt.figure(figsize=(14, 10))
    
    # 价格图表
    plt.subplot(3, 1, 1)
    plt.plot(prices, 'g-', label='Close Price')
    plt.title(f'Price and RSI - {title}')
    plt.ylabel('Price')
    plt.legend()
    
    # RSI图表
    plt.subplot(3, 1, 2)
    plt.plot(rsi, 'b-', label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.ylim(0, 100)
    plt.ylabel('RSI')
    plt.legend()
    
    # 交易信号图表
    plt.subplot(3, 1, 3)
    # 买入信号（绿色向上箭头）
    buy_indices = [i for i, signal in enumerate(buy_signals) if signal == 1]
    plt.plot(buy_indices, [prices[i] for i in buy_indices], 'g^', markersize=10, label='Buy Signal')
    
    # 卖出信号（红色向下箭头）
    sell_indices = [i for i, signal in enumerate(sell_signals) if signal == 1]
    plt.plot(sell_indices, [prices[i] for i in sell_indices], 'rv', markersize=10, label='Sell Signal')
    
    plt.plot(prices, 'k-', alpha=0.3, label='Close Price')
    plt.xlabel('Time')
    plt.ylabel('Price with Signals')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'test_results_{title.replace(" ", "_")}.png')
    plt.show()

def plot_trading_results(prices, rsi, buy_signals, sell_signals, portfolio_values, title):
    """绘制交易策略表现"""
    plt.figure(figsize=(16, 12))
    
    # 价格和交易信号
    plt.subplot(3, 1, 1)
    plt.plot(prices, 'g-', label='Close Price')
    
    # 买入信号（绿色向上箭头）
    buy_indices = [i for i, signal in enumerate(buy_signals) if signal == 1]
    plt.plot(buy_indices, [prices[i] for i in buy_indices], 'g^', markersize=8, label='Buy Signal')
    
    # 卖出信号（红色向下箭头）
    sell_indices = [i for i, signal in enumerate(sell_signals) if signal == 1]
    plt.plot(sell_indices, [prices[i] for i in sell_indices], 'rv', markersize=8, label='Sell Signal')
    
    plt.title(f'Trading Strategy Performance - {title}')
    plt.ylabel('Price')
    plt.legend()
    
    # RSI指标
    plt.subplot(3, 1, 2)
    plt.plot(rsi, 'b-', label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.ylim(0, 100)
    plt.ylabel('RSI')
    plt.legend()
    
    # 资产价值变化
    plt.subplot(3, 1, 3)
    plt.plot(portfolio_values, 'b-', label='Portfolio Value')
    plt.axhline(y=1000000, color='r', linestyle='--', label='Initial Capital')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'trading_results_{title.replace(" ", "_")}.png')
    plt.show()

def main():
    """主测试函数"""
    print("Starting Factor Module Tests with Simulated Data...")
    
    # 测试因子性能
    test_factor_performance()
    
    # 测试大规模数据处理
    test_large_scale()
    
    # 测试交易策略
    test_trading_strategy()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
