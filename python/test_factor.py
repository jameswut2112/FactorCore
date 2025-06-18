import sys
import os
import numpy as np
import pandas as pd

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
    from factors import CustomFactor
    print("成功导入 C++ 模块")
    
    # 导入包装器模块（使用绝对导入）
    from factor_wrapper import PyFactor
    print("成功导入 Python 包装器")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)
    
    

def test_custom_factor():
    # 创建测试数据
    np.random.seed(42)
    data = np.random.rand(100, 3)  # 100行3列
    df = pd.DataFrame(data, columns=['open', 'high', 'low'])
    
    # 使用C++因子
    cpp_factor = CustomFactor()
    
    # 使用Python包装器
    py_factor = PyFactor(CustomFactor)
    
    # 测试计算
    try:
        print("测试C++直接计算...")
        result_cpp = cpp_factor.calculate(df.values, np.array([5.0]))
        print(f"C++结果 (前5个): {result_cpp[:5]}")
        
        print("\n测试Python包装器...")
        result_py = py_factor.calculate(df, window=5)
        print(f"Python包装器结果 (前5个):\n{result_py.head()}")
        
        # 验证结果一致性
        assert np.allclose(result_cpp, result_py.values, equal_nan=True), "结果不一致"
        print("\n测试通过: C++和Python包装器结果一致")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_factor()
