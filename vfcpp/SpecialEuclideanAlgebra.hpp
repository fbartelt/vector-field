#ifndef SPECIAL_EUCLIDEAN_ALGEBRA_HPP
#define SPECIAL_EUCLIDEAN_ALGEBRA_HPP

#include <eigen3/Eigen/Dense>

#include "GeneralLinearAlgebra.hpp"

// Upper bound for theta approximately zero for se3::exp
float c_theta_zero = 1e-6;

class SpecialEuclideanAlgebra
    : public GeneralLinearAlgebraBase<SpecialEuclideanAlgebra> {
 public:
  SpecialEuclideanAlgebra(int n)
      : GeneralLinearAlgebraBase<SpecialEuclideanAlgebra>(n + 1) {
    n_ = n + 1;
    dim_ = n_ * (n_ - 1) / 2;
    signature_ = "(" + std::to_string(n_ - 1) + ",1)";
    name_ = "se";
  }

  SpecialEuclideanAlgebra(const Eigen::MatrixXd& matrix)
      : GeneralLinearAlgebraBase<SpecialEuclideanAlgebra>(matrix.rows()) {
    n_ = matrix.rows();
    dim_ = n_ * (n_ - 1) / 2;
    signature_ = "(" + std::to_string(n_ - 1) + ")";
    name_ = "se";
  }

  SpecialEuclideanAlgebra S(const Eigen::VectorXd& xi) const override;
  Eigen::VectorXd invS(const Eigen::MatrixXd& A) const override;
  Eigen::VectorXd invS(const SpecialEuclideanAlgebra& A) const {
    return invS(A.matrix_);
  }
};

SpecialEuclideanAlgebra SpecialEuclideanAlgebra::S(
    const Eigen::VectorXd& xi) const {
  // Check if the dimension of xi matches the Lie algebra dimension
  if (xi.size() != dim_) {
    std::string msg = "Incorrect dimension for xi. Expected " +
                      std::to_string(dim_) + " but got " +
                      std::to_string(xi.size());
    throw std::invalid_argument(msg);
  }

  // Create an nxn matrix filled with zeros using Eigen
  Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(n_, n_);

  // Get the indices for the upper triangular part of the matrix (excluding the
  // diagonal) Portion of Lie algebra related to SO(n)
  int index = dim_ - 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = i + 1; j < n_; ++j) {
      S_(i, j) = pow((-1), index) * xi(index);
      index--;
    }
  }

  S_ = S_ - S_.transpose().eval();

  // Portion of Lie algebra related to R^n
  for (int i = 0; i < n_; ++i) {
    S_(i, n_) = xi(i);
  }

  return SpecialEuclideanAlgebra(S_);
}

Eigen::VectorXd SpecialEuclideanAlgebra::invS(const Eigen::MatrixXd& A) const {
  if (A.rows() != n_ || A.cols() != n_) {
    std::string msg = "Incorrect dimension for A. Expected " +
                      std::to_string(n_) + "x" + std::to_string(n_) +
                      " but got " + std::to_string(A.rows()) + "x" +
                      std::to_string(A.cols());
    throw std::invalid_argument(msg);
  }

  Eigen::VectorXd xi = Eigen::VectorXd::Zero(dim_);

  // Get the indices for the upper triangular part of the matrix (excluding the
  // diagonal) Portion of Lie algebra related to SO(n)
  int index = dim_ - 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = i + 1; j < n_; ++j) {
      // Emulates the basis for SE(3) -- upper triangular as -z y; -x. So every
      // odd xi_i is negated and the matrix is populated in reverse order
      xi(index) = pow((-1), index) * A(i, j);
      index--;
    }
  }

  // Portion of Lie algebra related to R^n
  for (int i = 0; i < n_; ++i) {
    xi(i) = A(i, n_);
  }

  return xi;
}

class se3 : public GeneralLinearAlgebraBase<se3> {
 public:
  se3() : GeneralLinearAlgebraBase<se3>(4) {
    n_ = 4;
    dim_ = 6;
    signature_ = "(3)";
    name_ = "se";
  }

  se3(const Eigen::MatrixXd& matrix)
      : GeneralLinearAlgebraBase<se3>(matrix.rows()) {
    if (matrix.rows() != 4 || matrix.cols() != 4) {
      std::string msg = "Incorrect dimension for A. Expected 4x4 but got " +
                        std::to_string(matrix.rows()) + "x" +
                        std::to_string(matrix.cols());
      throw std::invalid_argument(msg);
    }
    n_ = 4;
    dim_ = 6;
    signature_ = "(3)";
    name_ = "se";
  }

  se3(const SpecialEuclideanAlgebra& A) : GeneralLinearAlgebraBase<se3>(4) {
    Eigen::MatrixXd mat = A.matrix();
    if (mat.rows() != 4 || mat.cols() != 4) {
      std::string msg = "Incorrect dimension for A. Expected 4x4 but got " +
                        std::to_string(mat.rows()) + "x" +
                        std::to_string(mat.cols());
      throw std::invalid_argument(msg);
    }
    n_ = 4;
    dim_ = 6;
    signature_ = "(3)";
    name_ = "se";
    matrix_ = mat;
  }

  se3 S(const Eigen::VectorXd& xi) const override {}

  Eigen::VectorXd invS(const Eigen::MatrixXd& A) const override {}

  Eigen::VectorXd invS(const se3& A) const { return se3::invS(A); }
  Eigen::VectorXd invS(const SpecialEuclideanAlgebra& A) const {
    se3 B = se3(A);
    return se3::invS(B);
  }

  Eigen::MatrixXd exp() const override {}
};

se3 se3::S(const Eigen::VectorXd& xi) const {
  if (xi.size() != 6) {
    std::string msg = "Incorrect dimension for xi. Expected 6 but got " +
                      std::to_string(xi.size());
    throw std::invalid_argument(msg);
  }

  Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(4, 4);

  // Portion of Lie algebra related to SO(n)
  S_(0, 1) = -xi(5);
  S_(0, 2) = xi(4);
  S_(1, 2) = -xi(3);
  S_ = S_ - S_.transpose().eval();

  S_(0, 3) = xi(0);
  S_(1, 3) = xi(1);
  S_(2, 3) = xi(2);

  return se3(S_);
}

Eigen::VectorXd se3::invS(const Eigen::MatrixXd& A) const {
  if (A.rows() != 4 || A.cols() != 4) {
    std::string msg = "Incorrect dimension for A. Expected 4x4 but got " +
                      std::to_string(A.rows()) + "x" + std::to_string(A.cols());
    throw std::invalid_argument(msg);
  }

  Eigen::VectorXd xi = Eigen::VectorXd::Zero(6);

  xi(0) = A(0, 3);
  xi(1) = A(1, 3);
  xi(2) = A(2, 3);
  xi(3) = -A(1, 2);
  xi(4) = A(0, 2);
  xi(5) = -A(0, 1);

  return xi;
}

Eigen::MatrixXd se3::exp() const {
  Eigen::Matrix3d A = matrix_.block<3, 3>(0, 0);
  Eigen::Vector3d v = matrix_.block<3, 1>(0, 3);
  float theta = sqrt(pow(A(1, 0), 2) + pow(A(0, 2), 2) + pow(A(2, 1), 2));
  // If theta is close to zero, use the first order approximation
  if (theta < c_theta_zero) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.block<3, 3>(0, 0) = R;
    result.block<3, 1>(0, 3) = v;
    return result;
  } else {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(theta) / theta) * A +
                        ((1 - cos(theta)) / pow(theta, 2)) * A * A;
    Eigen::Matrix3d U = Eigen::Matrix3d::Identity() +
                        ((1 - cos(theta)) / pow(theta, 2)) * A +
                        ((theta - sin(theta)) / pow(theta, 3)) * A * A;
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.block<3, 3>(0, 0) = R;
    result.block<3, 1>(0, 3) = U * v;
    return result;
  }
}

using seN = SpecialEuclideanAlgebra;

#endif  // SPECIAL_EUCLIDEAN_ALGEBRA_HPP