#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "factor_base.hpp"
#include "custom_factor.hpp"  // 包含具体因子

namespace py = pybind11;

// 模块入口点
PYBIND11_MODULE(factors, m) {
    m.doc() = "C++ model";

    // 暴露基类
    py::class_<FactorBase>(m, "FactorBase")
        .def("calculate", &FactorBase::calculate)
        .def("validate_params", &FactorBase::validate_params);

    // 暴露具体因子类
    py::class_<CustomFactor, FactorBase>(m, "CustomFactor")
        .def(py::init<>());
}
