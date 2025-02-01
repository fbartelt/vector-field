#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <eigen3/Eigen/Dense>
#include <string>

#include "include/GeneralLinearAlgebra.hpp"
#include "include/GeneralLinearGroup.hpp"
#include "include/LieAlgebra.hpp"
#include "include/LieGroup.hpp"
#include "include/ProperOrthochronousLorentzAlgebra.hpp"
#include "include/ProperOrthochronousLorentzGroup.hpp"
#include "include/SpecialEuclideanAlgebra.hpp"
#include "include/SpecialEuclideanGroup.hpp"
#include "include/VectorField.hpp"

namespace py = pybind11;

// cd build
// cmake ..
// cmake --build .

template <typename Group, typename Algebra>
void bindLieGroup(py::module& m, const std::string& name) {
  py::class_<Group>(m, name.c_str())
      .def(py::init<>())                        // Default constructor
      .def(py::init<const Eigen::MatrixXd&>())  // Constructor with matrix
      .def(py::init<const Group&>())            // Copy constructor
      .def("dim", &Group::dim)
      .def("n", &Group::n)
      .def("matrix", &Group::matrix)
      .def("setMatrix", &Group::setMatrix)
      .def("print", &Group::print)
      .def("random", &Group::random)
      .def("algebra", &Group::algebra)
      .def(py::self + py::self)
      .def(
          "__add__",
          [](const Group& self, const Eigen::MatrixXd& other) {
            return self + other;
          },
          py::is_operator())
      .def(
          "__sub__",
          [](const Group& self, const Eigen::MatrixXd& other) {
            return self - other;
          },
          py::is_operator())
      .def(py::self - py::self)
      .def(-py::self)
      .def(py::self * py::self)
      .def(
          "__mul__",
          [](const Group& self, const Eigen::MatrixXd& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const Eigen::MatrixXd& other, const Group& self) {
            return other * self;
          },
          py::is_operator())
      .def(
          "__matmul__",
          [](const Group& self, const Eigen::MatrixXd& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__matmul__",
          [](const Eigen::MatrixXd& other, const Group& self) {
            return other * self;
          },
          py::is_operator())
      .def(
          "__matmul__",
          [](const Group& self, const Group& other) { return self * other; },
          py::is_operator())
      .def(float() * py::self)
      .def(py::self * float())
      .def(py::self / float())
      .def(
          "__add__",
          [](const Eigen::MatrixXd& other, const Group& self) {
            return other + self;
          },
          py::is_operator())
      .def(
          "__sub__",
          [](const Eigen::MatrixXd& other, const Group& self) {
            return other - self;
          },
          py::is_operator())
      .def("__repr__", &Group::repr);
}

template <typename Algebra>
void bindLieAlgebra(py::module& m, const std::string& name) {
  py::class_<Algebra>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<const Eigen::MatrixXd&>())
      .def(py::init<const Algebra&>())
      .def("dim", &Algebra::dim)
      .def("matrix", &Algebra::matrix)
      .def("setMatrix", &Algebra::setMatrix)
      .def("print", &Algebra::print)
      .def("exp", &Algebra::exp)
      .def(py::self * float())
      .def(
          "__mul__",
          [](const Algebra& self, const Eigen::VectorXd& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__matmul__",
          [](const Algebra& self, const Eigen::VectorXd& other) {
            return self * other;
          },
          py::is_operator())
      .def("S", &Algebra::S)
      .def("invS", py::overload_cast<const Eigen::MatrixXd&>(&Algebra::invS,
                                                             py::const_))
      .def("invS",
           py::overload_cast<const Algebra&>(&Algebra::invS, py::const_))
      .def("__repr__", &Algebra::repr);
}

template <typename Group>
void bindVectorField(py::module& m, const std::string& name) {
  py::class_<VectorField<Group>>(m, name.c_str())
      .def(py::init<const std::vector<Eigen::MatrixXd>&, double, double>(),
           py::arg("curve"), py::arg("delta") = 0.001, py::arg("ds_") = 0.001)
      .def(py::init<const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&, double, double>(),
           py::arg("curve"), py::arg("curve_derivative"),
           py::arg("delta") = 0.001, py::arg("ds_") = 0.001)
      .def_readwrite("iterationResults", &VectorField<Group>::iterationResults)
      .def_readwrite("norm_hist", &VectorField<Group>::norm_hist)
      .def_readwrite("closest_indices", &VectorField<Group>::closest_indices)
      .def("__call__", &VectorField<Group>::operator())
      .def("kronDelta", &VectorField<Group>::kronDelta)
      .def("EEdistance", &VectorField<Group>::EEdistance)
      .def("ECdistance", &VectorField<Group>::ECdistance)
      .def("eval", &VectorField<Group>::eval, py::arg("state"),
           py::arg("save_data") = false, py::arg("gain_n1") = 1.0,
           py::arg("gain_n2") = 1.0, py::arg("gain_t1") = 1.0,
           py::arg("gain_t2") = 1.0, py::arg("gain_t3") = 1.0)
      .def("EEdistSE3Variables", &VectorField<Group>::EEdistSE3Variables)
      .def("Shat", &VectorField<Group>::Shat);
}

PYBIND11_MODULE(_vectorfield, m) {
  bindLieGroup<SE3, se3>(m, "SE3");
  // bindLieGroup<SEN, seN>(m, "SpecialEuclideanGroup"); // NEEDS ANOTHER BINDER
  // SINCE INIT TAKES "N" AS ARGUMENT
  bindLieGroup<SO31, so31>(m, "SO31");
  // bindLieGroup<GeneralLinearGroup, GeneralLinearAlgebra>(m,
  // "GeneralLinearGroup"); // NEEDS ANOTHER BINDER SINCE INIT TAKES "N" AS
  // ARGUMENT
  bindLieAlgebra<se3>(m, "se3");
  bindLieAlgebra<so31>(m, "so31");
  // bindLieAlgebra<seN>(m, "SpecialEuclideanAlgebra"); // NEEDS ANOTHER BINDER
  // SINCE INIT TAKES "N" AS ARGUMENT bindLieAlgebra<GeneralLinearAlgebra>(m,
  // "GeneralLinearAlgebra"); // NEEDS ANOTHER BINDER SINCE INIT TAKES "N" AS
  // ARGUMENT

  bindVectorField<SE3>(m, "_VectorFieldSE3");
  // bindVectorField<SEN>(m, "_VectorFieldSEN");
  bindVectorField<SO31>(m, "_VectorFieldSO31");
  // bindVectorField<GeneralLinearGroup>(m, "_VectorFieldGL");

  #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
  #else
    m.attr("__version__") = "dev";
  #endif
}