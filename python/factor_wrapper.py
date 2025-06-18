import numpy as np
import pandas as pd


# 使用绝对导入
try:
    from factors import FactorBase
except ImportError:
    # 回退方案
    import sys
    import os
    # build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "bin", "Release")
    build_dir = r"D:\project_C++_FactorCore\FactorCore\out\build\x64-Debug\bin"
    sys.path.insert(0, os.path.abspath(build_dir))
    from factors import FactorBase
    
    
class PyFactor:
    def __init__(self, cpp_factor_class):
        self.cpp_factor = cpp_factor_class()
        
    def calculate(self, df: pd.DataFrame, **params):
        # 转换pandas数据到NumPy数组
        cpp_data = df.values.astype(np.float64)
        
        # 准备参数向量
        param_keys = sorted(params.keys())
        cpp_params = np.array([params[k] for k in param_keys], dtype=np.float64)
        
        try:
            # 调用C++实现
            result = self.cpp_factor.calculate(cpp_data, cpp_params)
            
            # 转换回pandas Series
            return pd.Series(result, index=df.index, name='factor_value')
            
        except Exception as e:
            # 处理C++异常
            print(f"C++计算错误: {str(e)}")
            return pd.Series(np.nan, index=df.index, name='factor_error')
