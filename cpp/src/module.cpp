#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "factor_base.hpp"
#include "custom_factor.hpp"  // ������������

namespace py = pybind11;

// ģ����ڵ�
PYBIND11_MODULE(factors, m) {
    m.doc() = "C++ model";

    // ��¶����
    py::class_<FactorBase>(m, "FactorBase")
        .def("calculate", &FactorBase::calculate)
        .def("validate_params", &FactorBase::validate_params);

    // ��¶����������
    py::class_<CustomFactor, FactorBase>(m, "CustomFactor")
        .def(py::init<>());
}
