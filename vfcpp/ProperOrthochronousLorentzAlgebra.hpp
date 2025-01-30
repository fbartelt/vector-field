#pragma once

#include <eigen3/Eigen/Dense>

#include "GeneralLinearAlgebra.hpp"

class ProperOrthochronousLorentzAlgebra
    : public GeneralLinearAlgebraBase<ProperOrthochronousLorentzAlgebra> {
 private:
  int p_ = 3;
  int q_ = 1;

 public:
  // Constructor
  ProperOrthochronousLorentzAlgebra() : GeneralLinearAlgebraBase(4) {
    n_ = 4;
    dim_ = 6;
    signature_ = "(3,1)";
    name_ = "so⁺";
  }

  ProperOrthochronousLorentzAlgebra(const Eigen::MatrixXd& matrix)
      : GeneralLinearAlgebraBase(4) {
    n_ = 4;
    dim_ = 6;
    signature_ = "(3,1)";
    matrix_ = matrix;
    name_ = "so⁺";
  }

  ProperOrthochronousLorentzAlgebra S(const Eigen::VectorXd& xi) const override;
  Eigen::VectorXd invS(const Eigen::MatrixXd& A) const override;
  Eigen::VectorXd invS(const ProperOrthochronousLorentzAlgebra& A) const {
    return invS(A.matrix_);
  }
};

ProperOrthochronousLorentzAlgebra ProperOrthochronousLorentzAlgebra::S(
    const Eigen::VectorXd& xi) const {
  // Check if the dimension of xi matches the Lie algebra dimension
  if (xi.size() != dim_) {
    throw std::invalid_argument("xi has the wrong dimension.");
  }

  // Create an nxn matrix filled with zeros using Eigen
  Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(n_, n_);

  S_(0, 1) = -xi(2);
  S_(0, 2) = xi(1);
  S_(1, 2) = -xi(0);
  S_ = S_ - S_.transpose().eval();

  S_(0, 3) = xi(3);
  S_(1, 3) = xi(4);
  S_(2, 3) = xi(5);
  S_(3, 0) = xi(3);
  S_(3, 1) = xi(4);
  S_(3, 2) = xi(5);

  return ProperOrthochronousLorentzAlgebra(S_);
}

Eigen::VectorXd ProperOrthochronousLorentzAlgebra::invS(
    const Eigen::MatrixXd& A) const {
  if (A.rows() != n_ || A.cols() != n_) {
    throw std::invalid_argument("A has the wrong dimension.");
  }

  Eigen::VectorXd xi = Eigen::VectorXd::Zero(dim_);

  xi(0) = -A(1, 2);
  xi(1) = A(0, 2);
  xi(2) = -A(0, 1);
  xi(3) = A(0, 3);
  xi(4) = A(1, 3);
  xi(5) = A(2, 3);

  return xi;
};

using so31 = ProperOrthochronousLorentzAlgebra;