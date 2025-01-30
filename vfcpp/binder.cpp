#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <eigen3/Eigen/Dense>
#include <string>

#include "GeneralLinearAlgebra.hpp"
#include "GeneralLinearGroup.hpp"
#include "LieAlgebra.hpp"
#include "LieGroup.hpp"
#include "ProperOrthochronousLorentzAlgebra.hpp"
#include "ProperOrthochronousLorentzGroup.hpp"
#include "SpecialEuclideanAlgebra.hpp"
#include "SpecialEuclideanGroup.hpp"
#include "VectorField.hpp"

namespace py = pybind11;

// cd build
// cmake ..
// cmake --build .

template <typename Group, typename Algebra>
void bindLieGroup(py::module& m, const std::string& name) {
  py::class_<Group, GeneralLinearGroupBase<Group, Algebra>>(m, name.c_str())
      .def(py::init<>())                        // Default constructor
      .def(py::init<const Eigen::MatrixXd&>())  // Constructor with matrix
      .def(py::init<const Group&>())            // Copy constructor
      .def("dim", &Group::dim)
      .def("n", &Group::n)
      .def("matrix", &Group::matrix)
      .def("setMatrix", &Group::setMatrix)
      .def("print", &Group::print)
      .def("random", &Group::random)
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
  py::class_<Algebra, GeneralLinearAlgebraBase<Algebra>>(m, name.c_str())
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
      .def("S", &Algebra::S)
      .def("invS", py::overload_cast<const Eigen::MatrixXd&>(&Algebra::invS,
                                                             py::const_))
      .def("invS",
           py::overload_cast<const Algebra&>(&Algebra::invS, py::const_));
}

template <typename DerivedGroup, typename DerivedAlgebra>
void bindGeneralLinearGroupBase(py::module& m, const std::string& name) {
  py::class_<GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>>(m,
                                                                   name.c_str())
      .def(py::init<const DerivedAlgebra&>())  //
      .def("dim", &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::dim)
      .def("n", &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::n)
      .def("matrix",
           &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::matrix)
      .def("setMatrix",
           &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::setMatrix)
      .def("print",
           &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::print)
      .def("random",
           &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::random)
      .def(
          "__add__",
          [](const DerivedGroup& self, const DerivedGroup& other) {
            return self + other;
          },
          py::is_operator())
      .def(
          "__add__",
          [](const Eigen::MatrixXd& other, const DerivedGroup& self) {
            return other + self;
          },
          py::is_operator())
      .def(
          "__sub__",
          [](const DerivedGroup& self, const Eigen::MatrixXd& other) {
            return self - other;
          },
          py::is_operator())
      .def(
          "__sub__",
          [](const Eigen::MatrixXd& other, const DerivedGroup& self) {
            return other - self;
          },
          py::is_operator())
      .def(
          "__neg__", [](const DerivedGroup& self) { return -self; },
          py::is_operator())
      .def(
          "__mul__",
          [](const DerivedGroup& self, const DerivedGroup& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const DerivedGroup& self, const Eigen::MatrixXd& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const Eigen::MatrixXd& self, const DerivedGroup& other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const Eigen::MatrixXd& self, const float other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const float other, const Eigen::MatrixXd& self) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__truediv__",
          [](const Eigen::MatrixXd& self, const float other) {
            return self / other;
          },
          py::is_operator())
      .def("__repr__",
           &GeneralLinearGroupBase<DerivedGroup, DerivedAlgebra>::print);
}

template <typename DerivedAlgebra>
class PyLieAlgebra : public LieAlgebra<DerivedAlgebra> {
public:
    /* Inherit the constructors */
    using LieAlgebra<DerivedAlgebra>::LieAlgebra;

    /* Trampoline (need one for each virtual function) */
    DerivedAlgebra S(const Eigen::VectorXd& xi) const override {
        PYBIND11_OVERRIDE_PURE(
            DerivedAlgebra, /* Return type */
            LieAlgebra<DerivedAlgebra>,      /* Parent class */
            S,          /* Name of function in C++ (must match Python name) */
            xi      /* Argument(s) */
        );
    }
    Eigen::VectorXd invS(const Eigen::MatrixXd& matrix) const override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::VectorXd, 
            LieAlgebra<DerivedAlgebra>,
            invS,
            matrix
        );
    }
    Eigen::MatrixXd exp() const override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, 
            LieAlgebra<DerivedAlgebra>,
            exp
        );
    }
};

template <typename DerivedAlgebra>
void bindAbstractLieAlgebra(py::module& m, const std::string& name){
    py::class_<LieAlgebra<DerivedAlgebra>, PyLieAlgebra<DerivedAlgebra>>(m, name.c_str())
        .def(py::init<>())
        .def("exp", &LieAlgebra<DerivedAlgebra>::exp)
        .def("S", &LieAlgebra<DerivedAlgebra>::S)
        .def("invS", &LieAlgebra<DerivedAlgebra>::invS);
}


template <typename DerivedAlgebra>
void bindGeneralLinearAlgebraBase(py::module& m, const std::string& name) {
  py::class_<GeneralLinearAlgebraBase<DerivedAlgebra>>(m, name.c_str())
      .def(py::init<const Eigen::MatrixXd&>())  //
      .def("dim", &GeneralLinearAlgebraBase<DerivedAlgebra>::dim)
      .def("matrix", &GeneralLinearAlgebraBase<DerivedAlgebra>::matrix)
      .def("setMatrix", &GeneralLinearAlgebraBase<DerivedAlgebra>::setMatrix)
      .def("print", &GeneralLinearAlgebraBase<DerivedAlgebra>::print)
      .def(
          "__mul__",
          [](const DerivedAlgebra& self, const float other) {
            return self * other;
          },
          py::is_operator())
      .def(
          "__mul__",
          [](const DerivedAlgebra& self, const Eigen::VectorXd& other) {
            return self * other;
          },
          py::is_operator())
      .def("exp", &GeneralLinearAlgebraBase<DerivedAlgebra>::exp)
      .def("S", &GeneralLinearAlgebraBase<DerivedAlgebra>::S)
      .def("invS", &GeneralLinearAlgebraBase<DerivedAlgebra>::invS)
      .def("__repr__", &GeneralLinearAlgebraBase<DerivedAlgebra>::repr);
}

PYBIND11_MODULE(pylie, m) {
  // bind_lie_group<se3>(m, "LieGroupSE3");
  // bind_lie_group<seN>(m, "LieGroupSEN");
  // bind_lie_group<so3>>(m, "LieGroupSO31");
  // bind_lie_group<GeneralLinearAlgebra>(m, "LieGroupGL");

  // bind_lie_algebra<se3>(m, "LieAlgebrase3");
  // bind_lie_algebra<seN>(m, "LieAlgebraseN");
  // bind_lie_algebra<GeneralLinearAlgebra>(m, "LieAlgebraGL");
  // bind_lie_algebra<so31>(m, "LieAlgebraso31");
  bindAbstractLieAlgebra<se3>(m, "LieAlgebrase3");
  bindGeneralLinearGroupBase<SE3, se3>(m, "BaseSE3");
  bindGeneralLinearAlgebraBase<se3>(m, "Basese3");

  bindLieGroup<SE3, se3>(m, "SE3");
  bindLieAlgebra<se3>(m, "se3");

  // py::class_<SE3, GeneralLinearGroupBase<SE3, se3>>(m, "SE3")
  //     .def(py::init<>())  // Default constructor
  //     .def(py::init<const Eigen::MatrixXd&>())  // Constructor with matrix
  //     .def(py::init<const SE3&>())  // Copy constructor
  //     .def("dim", &SE3::dim)
  //     .def("n", &SE3::n)
  //     .def("matrix", &SE3::matrix)
  //     .def("setMatrix", &SE3::setMatrix)
  //     .def("print", &SE3::print)
  //     .def("random", &SE3::random)
  //     .def(py::self + py::self)
  //     .def("__add__", [](const SE3& self, const Eigen::MatrixXd& other) {
  //         return self + other;
  //     }, py::is_operator())
  //     .def("__sub__", [](const SE3& self, const Eigen::MatrixXd& other) {
  //         return self - other;
  //     }, py::is_operator())
  //     .def(py::self - py::self)
  //     .def(-py::self)
  //     .def(py::self * py::self)
  //     .def(float() * py::self)
  //     .def(py::self * float())
  //     .def(py::self / float())
  //     .def("__add__", [](const Eigen::MatrixXd& other, const SE3& self) {
  //         return other + self;
  //     }, py::is_operator())
  //     .def("__sub__", [](const Eigen::MatrixXd& other, const SE3& self) {
  //         return other - self;
  //     }, py::is_operator())
  //     .def("__repr__", &SE3::repr)

  // py::class_<se3, GeneralLinearAlgebraBase<se3>>(m, "se3")
  //     .def(py::init<>())
  //     .def(py::init<const Eigen::MatrixXd&>())
  //     .def(py::init<const se3&>())
  //     .def("dim", &se3::dim)
  //     .def("matrix", &se3::matrix)
  //     .def("setMatrix", &se3::setMatrix)
  //     .def("print", &se3::print)
  //     .def("exp", &se3::exp)
  //     .def(py::self * float())
  //     .def("__mul__", [](const se3& self, const Eigen::VectorXd& other) {
  //         return self * other;
  //     }, py::is_operator())
  //     .def("S", &se3::S)
  //     .def("invS", py::overload_cast<const Eigen::MatrixXd&>(&se3::invS,
  //     py::const_)) .def("invS", py::overload_cast<const se3&>(&se3::invS,
  //     py::const_));
}