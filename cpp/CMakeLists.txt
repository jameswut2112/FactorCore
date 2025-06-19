# 查找依赖
find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)

# 添加库目标
add_library(factors MODULE
    src/module.cpp
    src/factor_base.cpp
    src/custom_factor.cpp  # 添加具体因子实现
 "include/market_data.hpp")

# 设置模块名
set_target_properties(factors PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"  # Windows扩展名
)

# 包含目录
target_include_directories(factors PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    include  # 关键：添加本地include目录
)

# 链接库
target_link_libraries(factors PRIVATE
    pybind11::module
    Eigen3::Eigen
)

# 设置C++标准
set_target_properties(factors PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
