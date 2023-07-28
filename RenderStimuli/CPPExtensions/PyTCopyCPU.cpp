#include <pybind11/pybind11.h>

#include "TCopyCPU.h"

namespace py = pybind11;

PYBIND11_MODULE(PyTCopyCPU, m)
{
  m.doc() = "TCopyCPU Module";
  py::class_<TCopyCPU>(m, "TCopyCPU")
    .def(py::init<>())
    .def("process",
      &TCopyCPU::process);
}

