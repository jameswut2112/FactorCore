//#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
//#include "factor_base.hpp"
//#include "custom_factor.hpp"  // 包含具体因子
//
//namespace py = pybind11;
//
//// 模块入口点
//PYBIND11_MODULE(factors, m) {
//    m.doc() = "C++ model";
//
//    // 暴露基类
//    py::class_<FactorBase>(m, "FactorBase")
//        .def("calculate", &FactorBase::calculate)
//        .def("validate_params", &FactorBase::validate_params);
//
//    // 暴露具体因子类
//    py::class_<CustomFactor, FactorBase>(m, "CustomFactor")
//        .def(py::init<>());
//}



#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "market_data.hpp"
#include "factor_base.hpp"
#include "custom_factor.hpp"

namespace py = pybind11;

// 绑定MarketDataType枚举
PYBIND11_MODULE(factors, m) {
    // 绑定MarketDataType枚举
    py::enum_<MarketDataType>(m, "MarketDataType")
        .value("OPEN", MarketDataType::OPEN)
        .value("HIGH", MarketDataType::HIGH)
        .value("LOW", MarketDataType::LOW)
        .value("CLOSE", MarketDataType::CLOSE)
        .value("VOLUME", MarketDataType::VOLUME)
        .export_values();

    // 绑定MarketData类
    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("open", &MarketData::open)
        .def_readwrite("high", &MarketData::high)
        .def_readwrite("low", &MarketData::low)
        .def_readwrite("close", &MarketData::close)
        .def_readwrite("volume", &MarketData::volume)
        .def("validate", &MarketData::validate)
        .def("size", &MarketData::size)
        .def("get", &MarketData::get, py::arg("data_type"))
        .def("latest_open", &MarketData::latest_open)
        .def("latest_high", &MarketData::latest_high)
        .def("latest_low", &MarketData::latest_low)
        .def("latest_close", &MarketData::latest_close)
        .def("latest_volume", &MarketData::latest_volume)
        .def("slice", &MarketData::slice, py::arg("start"), py::arg("end"));

    // 绑定因子基类
    py::class_<FactorBase>(m, "FactorBase")
        .def("set_para_range", &FactorBase::setParaRange,
            py::arg("pmin"), py::arg("pmax"), py::arg("pstep"), py::arg("default_params"),
            "Set parameter ranges and default values")
        .def("get_pmin", &FactorBase::get_pmin, "Get minimum parameter values")
        .def("get_pmax", &FactorBase::get_pmax, "Get maximum parameter values")
        .def("get_pstep", &FactorBase::get_pstep, "Get parameter step sizes")
        .def("get_default", &FactorBase::get_default, "Get default parameter values")
        .def("eval", &FactorBase::eval,
            py::arg("market_data"), py::arg("params") = std::vector<int>(),
            "Evaluate factor and generate signals (uses smart update)")
        .def("get_value", &FactorBase::get_value,
            "Get current factor values");

    // 绑定TR0因子
    py::class_<TR0, FactorBase>(m, "TR0")
        .def(py::init<>(), "RSI-based trading factor");
}
