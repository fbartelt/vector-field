#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>

#include "GeneralLinearAlgebra.hpp"
#include "LieGroup.hpp"

template <typename DerivedGroup, typename DerivedAlgebra>
class GeneralLinearGroupBase : public LieGroup<DerivedAlgebra> {
 protected:
  int n_;                   // matrix dimension
  Eigen::MatrixXd matrix_;  // The matrix representation of SE(n)
  int dim_;                 // Dimension of the Lie algebra (n * (n - 1) / 2)
  std::string name_ = "GL";
  std::string signature_;

 public:
  GeneralLinearGroupBase(const DerivedAlgebra& algebra)
      : LieGroup<DerivedAlgebra>(algebra) {}
  GeneralLinearGroupBase(const GeneralLinearGroupBase& other)
      : LieGroup<DerivedAlgebra>(other.algebra_) {}

  int dim() const { return dim_; }
  int n() const { return n_; }
  const Eigen::MatrixXd& matrix() const { return matrix_; }
  void setMatrix(const Eigen::MatrixXd& matrix) { matrix_ = matrix; }

  std::string repr() const {
    std::ostringstream oss;
    oss << "Element of " << name_ << signature_ << std::endl << matrix_;
    return oss.str();
  }

  void print() const {
    std::cout << "Element of " << name_ << signature_ << std::endl
              << matrix_ << std::endl;
  }
  DerivedGroup& operator=(const DerivedGroup& other) {
    if (this != &other) {
      this->matrix_ = other.matrix_;
    }
    return *this;
  }
  Eigen::MatrixXd random() const override {
    // THIS ACCESS THE DERIVED CLASS ATTRIBUTES DIM AND MATRIX
    int dim = this->algebra_.dim();
    Eigen::VectorXd xi = Eigen::VectorXd::Random(dim);
    DerivedAlgebra A = this->algebra_.S(xi);
    Eigen::MatrixXd exp_A = A.exp();
    return exp_A;
  }
  Eigen::MatrixXd operator+(const Eigen::MatrixXd& other) const {
    int n = static_cast<const DerivedGroup*>(this)->n();
    if ((n != other.rows()) || (n != other.cols())) {
      throw std::invalid_argument("Matrix dimensions must match.");
    }
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return mat + other;
  }
  Eigen::MatrixXd operator+(const DerivedGroup& other) const {
    Eigen::MatrixXd mat = static_cast<const DerivedGroup&>(other).matrix();
    return *this + mat;
  }

  Eigen::MatrixXd operator-() const {
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return -mat;
  }

  Eigen::MatrixXd operator-(const Eigen::MatrixXd& other) const {
    int n = static_cast<const DerivedGroup*>(this)->n();
    if ((n != other.rows()) || (n != other.cols())) {
      throw std::invalid_argument("Matrix dimensions must match.");
    }
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return mat - other;
  }

  Eigen::MatrixXd operator-(const DerivedGroup& other) const {
    return *this - other.matrix();
  }

  Eigen::MatrixXd operator*(const Eigen::MatrixXd& other) const {
    int n = static_cast<const DerivedGroup*>(this)->n();
    if ((n != other.rows()) || (n != other.cols())) {
      throw std::invalid_argument("Matrix dimensions must match.");
    }
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return mat * other.matrix();
  }
  DerivedGroup operator*(const DerivedGroup& other) const {
    int n = static_cast<const DerivedGroup*>(this)->n();
    if (n != other.n()) {
      throw std::invalid_argument("Matrix dimensions must match.");
    }
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    Eigen::MatrixXd result = mat * other.matrix();
    return DerivedGroup(result);
  }

  Eigen::MatrixXd operator*(const float scalar) const {
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return scalar * mat;
  }

  Eigen::MatrixXd operator/(const float scalar) const {
    Eigen::MatrixXd mat = static_cast<const DerivedGroup*>(this)->matrix();
    return mat / scalar;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const DerivedGroup& element) {
    os << element.matrix_;
    return os;
  }

  friend Eigen::MatrixXd operator+(const Eigen::MatrixXd& mat,
                                   const DerivedGroup& element) {
    Eigen::MatrixXd mat2 = static_cast<const DerivedGroup&>(element).matrix();
    return mat + mat2;
  }
  friend Eigen::MatrixXd operator-(const Eigen::MatrixXd& mat,
                                   const DerivedGroup& element) {
    Eigen::MatrixXd mat2 = static_cast<const DerivedGroup&>(element).matrix();
    return mat - mat2;
  }
  friend Eigen::MatrixXd operator*(const Eigen::MatrixXd& mat,
                                   const DerivedGroup& element) {
    Eigen::MatrixXd mat2 = static_cast<const DerivedGroup&>(element).matrix();
    return mat * mat2;
  }
  friend Eigen::MatrixXd operator*(const float scalar,
                                   const DerivedGroup& element) {
    Eigen::MatrixXd mat = static_cast<const DerivedGroup&>(element).matrix();
    return scalar * mat;
  }
};

class GeneralLinearGroup
    : public GeneralLinearGroupBase<GeneralLinearGroup, GeneralLinearAlgebra> {
 public:
  // Constructor passing only the integer n (random matrix initialization)
  GeneralLinearGroup(int n)
      : GeneralLinearGroupBase<GeneralLinearGroup, GeneralLinearAlgebra>(
            GeneralLinearAlgebra(n)) {
    // Initialize the matrix as a random SE(n) element
    n_ = n;
    dim_ = n * n;
    matrix_ = random();
    signature_ = "(" + std::to_string(n) + ")";
  }

  // Constructor with both n and a matrix
  GeneralLinearGroup(const Eigen::MatrixXd& mat)
      : GeneralLinearGroupBase<GeneralLinearGroup, GeneralLinearAlgebra>(
            GeneralLinearAlgebra(mat.rows())) {
    int n = mat.rows();
    dim_ = n * n;
    n_ = n;
    matrix_ = mat;
    signature_ = "(" + std::to_string(n) + ")";
  }

  GeneralLinearGroup(const GeneralLinearGroup& other)
      : GeneralLinearGroupBase<GeneralLinearGroup, GeneralLinearAlgebra>(
            GeneralLinearAlgebra(other.n_)) {
    n_ = other.n_;
    matrix_ = other.matrix_;
    dim_ = other.dim_;
    signature_ = other.signature_;
  }
};