import sys
import os
import numpy as np
import pandas as pd



import numpy as np
import matplotlib.pyplot as plt


# 添加模块路径
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "bin", "Debug"))
# sys.path.append(f"D:\\project_C++_FactorCore\\FactorCore\\out\\build\\x64-Debug\\bin")
sys.path.append(r"D:\project_C++_FactorCore\FactorCore\build\debug\bin\debug")

print(f"Python 路径: {sys.path}")


# try:
    # from factors import CustomFactor
    # from factor_wrapper import PyFactor
# except ImportError as e:
    # print(f"导入错误: {e}")
    # sys.exit(1)

try:
    # 导入 C++ 模块
    import factors
    # from factors import CustomFactor
    print("成功导入 C++ 模块")
    
    # # 导入包装器模块（使用绝对导入）
    # from factor_wrapper import PyFactor
    # print("成功导入 Python 包装器")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)
    
    

# def test_custom_factor():
    # # 创建测试数据
    # np.random.seed(42)
    # data = np.random.rand(100, 3)  # 100行3列
    # df = pd.DataFrame(data, columns=['open', 'high', 'low'])
    
    # # 使用C++因子
    # cpp_factor = CustomFactor()
    
    # # 使用Python包装器
    # py_factor = PyFactor(CustomFactor)
    
    # # 测试计算
    # try:
        # print("测试C++直接计算...")
        # result_cpp = cpp_factor.calculate(df.values, np.array([5.0]))
        # print(f"C++结果 (前5个): {result_cpp[:5]}")
        
        # print("\n测试Python包装器...")
        # result_py = py_factor.calculate(df, window=5)
        # print(f"Python包装器结果 (前5个):\n{result_py.head()}")
        
        # # 验证结果一致性
        # assert np.allclose(result_cpp, result_py.values, equal_nan=True), "结果不一致"
        # print("\n测试通过: C++和Python包装器结果一致")
        
    # except Exception as e:
        # print(f"测试失败: {str(e)}")
        # import traceback
        # traceback.print_exc()

# if __name__ == "__main__":
    # test_custom_factor()












class StructuredData:
    def __init__(self, close=None, **kwargs):
        self.close = np.array(close) if close is not None else np.array([])
        self.open = kwargs.get('open', self.close.copy())
        self.high = kwargs.get('high', self.close.copy())
        self.low = kwargs.get('low', self.close.copy())
        self.vol = kwargs.get('vol', np.ones_like(self.close))

def sample_data(size=100):
    """生成简化版模拟数据"""
    np.random.seed(42)
    
    # 基础价格序列
    base_prices = np.cumprod(1 + np.random.normal(0, 0.015, size)) * 100
    
    # 生成其他价格数据
    opens = base_prices * (1 + np.random.normal(0, 0.005, size))
    highs = base_prices * (1 + np.random.uniform(0, 0.01, size))
    lows = base_prices * (1 - np.random.uniform(0, 0.01, size))
    
    # 成交量
    volumes = np.random.randint(10000, 50000, size)
    
    return StructuredData(
        close=base_prices,
        open=opens,
        high=highs,
        low=lows,
        vol=volumes
    )

def structured_to_market_data(structured_data):
    """添加严格的数据验证"""
    if len(structured_data.close) < 15:  # RSI默认period=14需要至少15个点
        raise ValueError(f"需要至少15个数据点，当前只有{len(structured_data.close)}个")
    
    # 检查所有数组长度一致


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
        return market_data
    except Exception as e:
        print(f"Market data validation failed: {str(e)}")
        return None


def structured_to_market_data(structured_data):
    """将StructuredData转换为MarketData"""
    # 添加长度一致性检查
    lengths = {
        'open': len(structured_data.open),
        'high': len(structured_data.high),
        'low': len(structured_data.low),
        'close': len(structured_data.close),
        'volume': len(structured_data.vol)
    }
    
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent data lengths: {lengths}")
    
    if lengths['close'] < 2:
        raise ValueError("Need at least 2 data points for calculation")
    
    market_data = factors.MarketData()
    market_data.open = structured_data.open.astype(np.float64)
    market_data.high = structured_data.high.astype(np.float64)
    market_data.low = structured_data.low.astype(np.float64)
    market_data.close = structured_data.close.astype(np.float64)
    market_data.volume = structured_data.vol.astype(np.float64)
    
    return market_data





def test_basic_functionality():
    """测试因子基本功能"""
    print("===== Testing Basic Factor Functionality =====")
    
    # 生成模拟数据
    print("Generating sample data...")
    structured_data = sample_data(100)
    market_data = structured_to_market_data(structured_data)
    
    if market_data is None:
        return
    
    # 创建因子
    print("Creating TR0 factor...")
    TR0 = factors.TR0()
    
    # 使用默认参数计算
    print("Calculating with default parameters...")
    buy, sell, close = TR0.eval(market_data)
    rsi_values = TR0.get_value()
    
    # 打印部分结果
    print("\nFirst 10 RSI values:")
    print(rsi_values[:10])
    
    print("\nLast 10 RSI values:")
    print(rsi_values[-10:])
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 价格图表
    plt.subplot(2, 1, 1)
    plt.plot(market_data.close, 'g-', label='Close Price')
    plt.title('Price and RSI')
    plt.ylabel('Price')
    plt.legend()
    
    # RSI图表
    plt.subplot(2, 1, 2)
    plt.plot(rsi_values, 'b-', label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.ylim(0, 100)
    plt.xlabel('Time')
    plt.ylabel('RSI')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('basic_test_results.png')
    plt.show()
    
    print("\nBasic functionality test completed. Results saved to 'basic_test_results.png'")

def test_incremental_update():
    """测试增量更新功能"""
    print("\n===== Testing Incremental Update =====")
    
    # 初始数据
    print("Generating initial data (50 points)...")
    structured_data = sample_data(10000)
    market_data = structured_to_market_data(structured_data)
    
    if market_data is None:
        return
    
    # 创建因子
    TR0 = factors.TR0()
    
    # 第一次计算（全量）
    print("First calculation (full)...")
    buy1, sell1, close1 = TR0.eval(market_data)
    rsi1 = TR0.get_value()
    print(f"Initial RSI values count: {len(rsi1)}")
    
    # 添加一个新数据点
    print("\nAdding one new data point...")
    new_close = market_data.close[-1] * (1 + np.random.normal(0, 0.01))
    new_open = new_close * (1 + np.random.normal(0, 0.005))
    new_high = new_close * (1 + np.random.uniform(0, 0.01))
    new_low = new_close * (1 - np.random.uniform(0, 0.01))
    new_vol = np.random.randint(10000, 50000)
    
    # 创建新市场数据
    new_structured_data = StructuredData(
        close=np.append(structured_data.close, new_close),
        open=np.append(structured_data.open, new_open),
        high=np.append(structured_data.high, new_high),
        low=np.append(structured_data.low, new_low),
        vol=np.append(structured_data.vol, new_vol)
    )
    new_market_data = structured_to_market_data(new_structured_data)
    
    # 第二次计算（增量更新）
    print("Second calculation (incremental update)...")
    buy2, sell2, close2 = TR0.eval(new_market_data)
    
    
    
    
    
    rsi2 = TR0.get_value()
    
    print(f"Updated RSI values count: {len(rsi2)}")
    print(f"Last RSI value: {rsi2[-1]}")
    
    # 验证增量更新结果
    if len(rsi2) == len(rsi1) + 1:
        print("✅ Incremental update successful: RSI array length increased by 1.")
    else:
        print("❌ Incremental update failed: RSI array length not increased by 1.")
    
    # 比较最后一个点的计算（如果可能）
    if len(rsi1) > 0:
        print(f"Previous last RSI: {rsi1[-1]}, New last RSI: {rsi2[-1]}")

if __name__ == "__main__":
    # 测试基本功能
    test_basic_functionality()
    
    # 测试增量更新
    test_incremental_update()
    
    print("\nAll tests completed!")




























