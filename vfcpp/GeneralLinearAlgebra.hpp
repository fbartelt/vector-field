#ifndef GENERAL_LINEAR_ALGEBRA_HPP
#define GENERAL_LINEAR_ALGEBRA_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <string>

#include "LieAlgebra.hpp"

template <typename DerivedAlgebra>
class GeneralLinearAlgebraBase : public LieAlgebra<DerivedAlgebra> {
 protected:
  int dim_;
  int n_;
  Eigen::MatrixXd matrix_;
  std::string name_ = "gl";
  std::string signature_ = "(n)";

 public:
  GeneralLinearAlgebraBase(int n)
      : n_(n), dim_(n * n), matrix_(Eigen::MatrixXd::Zero(n, n)) {}
  int dim() const { return dim_; }

  void print() const {
    std::cout << "Element of " << name_ << signature_ << ":" << std::endl
              << matrix_ << std::endl;
  }
  Eigen::MatrixXd matrix() const { return matrix_; }
  void setMatrix(const Eigen::MatrixXd& matrix) { matrix_ = matrix; }

  DerivedAlgebra operator*(const float scalar) const {
    Eigen::MatrixXd result =
        scalar * static_cast<const DerivedAlgebra*>(this)->matrix();
    return DerivedAlgebra(result);
  }
  Eigen::VectorXd operator*(const Eigen::VectorXd& v) const {
    return static_cast<const DerivedAlgebra*>(this)->matrix() * v;
  }
  Eigen::MatrixXd exp() const override {
    return static_cast<const DerivedAlgebra*>(this)->matrix().exp();
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const DerivedAlgebra& element) {
    os << element.matrix_;
    return os;
  }
};

// template <typename DerivedAlgebra = GeneralLinearAlgebra>
class GeneralLinearAlgebra
    : public GeneralLinearAlgebraBase<GeneralLinearAlgebra> {
  //  private:
  //   int dim_;  // manifold dimension
  //   int n_;    // matrix dimension
  //   Eigen::MatrixXd matrix_;
 public:
  // Constructor
  GeneralLinearAlgebra(int n)
      : GeneralLinearAlgebraBase<GeneralLinearAlgebra>(n) {
    // Manifold dimension for se Lie algebra
    n_ = n;
    dim_ = n * n;
    signature_ = "(" + std::to_string(n) + ")";
  }

  GeneralLinearAlgebra(const Eigen::MatrixXd& matrix)
      : GeneralLinearAlgebraBase<GeneralLinearAlgebra>(matrix.rows()) {
    n_ = matrix.rows();
    dim_ = n_ * n_;
    matrix_ = matrix;
    signature_ = "(" + std::to_string(n_) + ")";
  }

  // Dimension of the Lie algebra

  GeneralLinearAlgebra S(const Eigen::VectorXd& xi) const override {
    // Check if the dimension of xi matches the Lie algebra dimension
    if (xi.size() != dim_) {
      std::string msg = "Incorrect dimension for xi. Expected " +
                        std::to_string(dim_) + " but got " +
                        std::to_string(xi.size());
      throw std::invalid_argument(msg);
    }

    Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(n_, n_);
    // Fill the Lie algebra element in order
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
        S_(i, j) = xi(i * n_ + j);
      }
    }

    return GeneralLinearAlgebra(S_);
  }
  Eigen::VectorXd invS(const Eigen::MatrixXd& A) const override {
    if (A.rows() != n_ || A.cols() != n_) {
      std::string msg = "Incorrect dimension for A. Expected " +
                        std::to_string(n_) + "x" + std::to_string(n_) +
                        " but got " + std::to_string(A.rows()) + "x" +
                        std::to_string(A.cols());
      throw std::invalid_argument(msg);
    }

    Eigen::VectorXd xi = Eigen::VectorXd::Zero(dim_);

    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
        xi(i * n_ + j) = A(i, j);
      }
    }

    return xi;
  }

  Eigen::VectorXd invS(const GeneralLinearAlgebra& A) const {
    return invS(A.matrix_);
  }
};

#endif  // SPECIAL_EUCLIDEAN_ALGEBRA_HPP