# -*- coding: utf-8 -*-
import sys
import os
import ctypes
import traceback
import platform

print("=" * 50)
print("Python 环境诊断")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")
print(f"系统平台: {platform.platform()}")
print(f"系统架构: {platform.architecture()[0]}")
print(f"当前工作目录: {os.getcwd()}")
print(f"系统路径: {sys.path}")

print("\n" + "=" * 50)
print("尝试加载模块")
print("=" * 50)

# 模块路径 - 根据您的实际路径调整
# module_path = r"D:\project_C++_FactorCore\FactorCore\out\build\x64-Debug\bin\factors.pyd"
module_path = r"D:\project_C++_FactorCore\FactorCore\build\debug\bin\debug\factors.pyd"

if not os.path.exists(module_path):
    print(f"错误: 模块文件不存在 - {module_path}")
    exit(1)

print(f"模块路径: {module_path}")
print(f"文件大小: {os.path.getsize(module_path)} 字节")

try:
    # 尝试直接加载模块
    mod = ctypes.cdll.LoadLibrary(module_path)
    print("成功加载模块!")
except Exception as e:
    print(f"加载失败: {type(e).__name__}: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()
    
    # Windows 错误代码
    if hasattr(e, 'winerror') and e.winerror is not None:
        print(f"\nWindows 错误代码: {e.winerror}")
        # 常见错误代码解释
        error_codes = {
            126: "找不到依赖模块 (ERROR_MOD_NOT_FOUND)",
            193: "%1 不是有效的 Win32 应用程序 (通常表示架构不匹配)",
            127: "找不到指定的程序 (可能缺少依赖)",
            1114: "动态链接库(DLL)初始化例程失败"
        }
        if e.winerror in error_codes:
            print(f"可能原因: {error_codes[e.winerror]}")
    
    print("\n建议解决方案:")
    print("1. 检查模块依赖项 (使用 Dependency Walker)")
    print("2. 确保使用相同的构建配置 (Release/Debug)")
    print("3. 验证 Python 和扩展模块的 ABI 兼容性")
    print("4. 检查模块初始化代码中的错误")

print("\n" + "=" * 50)
print("尝试导入模块")
print("=" * 50)

# 添加模块路径到 sys.path
build_dir = os.path.dirname(module_path)
sys.path.insert(0, build_dir)

try:
    import factors
    print("成功导入模块!")
    print(f"模块属性: {dir(factors)}")
except Exception as e:
    print(f"导入失败: {type(e).__name__}: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()
