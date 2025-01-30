#ifndef SPECIAL_EUCLIDEAN_GROUP_HPP
#define SPECIAL_EUCLIDEAN_GROUP_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "GeneralLinearGroup.hpp"
#include "SpecialEuclideanAlgebra.hpp"

class SpecialEuclideanGroup
    : public GeneralLinearGroupBase<SpecialEuclideanGroup,
                                SpecialEuclideanAlgebra> {
 public:
  SpecialEuclideanGroup(int n)
      : GeneralLinearGroupBase<SpecialEuclideanGroup, SpecialEuclideanAlgebra>(
            SpecialEuclideanAlgebra(n)) {
    n_ = n + 1;
    dim_ = n * (n - 1) / 2 + n;
    name_ = "SE";
    signature_ = "(" + std::to_string(n) + ")";
    matrix_ = random();
  }

  // Constructor with both n and a matrix
  SpecialEuclideanGroup(const Eigen::MatrixXd& matrix)
      : GeneralLinearGroupBase<SpecialEuclideanGroup, SpecialEuclideanAlgebra>(
            SpecialEuclideanAlgebra(matrix.rows())) {
    n_ = matrix.rows();
    int n = n_ - 1;
    dim_ = n * (n - 1) / 2 + n;
    name_ = "SE";
    signature_ = "(" + std::to_string(n) + ")";
    matrix_ = matrix;
  }

  SpecialEuclideanGroup(const SpecialEuclideanGroup& other)
      : GeneralLinearGroupBase<SpecialEuclideanGroup, SpecialEuclideanAlgebra>(
            SpecialEuclideanAlgebra(other.matrix())) {
    n_ = other.n();
    dim_ = other.dim();
    matrix_ = other.matrix();
    name_ = other.name_;
    signature_ = other.signature_;
  }
};

class SE3 : public GeneralLinearGroupBase<SE3, se3> {
 public:
  SE3() : GeneralLinearGroupBase<SE3, se3>(se3()) {
    n_ = 4;
    dim_ = 6;
    name_ = "SE";
    signature_ = "(3)";
    matrix_ = random();
  }

  SE3(const Eigen::MatrixXd& matrix)
      : GeneralLinearGroupBase<SE3, se3>(se3(matrix)) {
    n_ = 4;
    dim_ = 6;
    name_ = "SE";
    signature_ = "(3)";
    matrix_ = matrix;
  }

  SE3(const SE3& other) : GeneralLinearGroupBase<SE3, se3>(se3(other.matrix())) {
    n_ = other.n();
    dim_ = other.dim();
    matrix_ = other.matrix();
    name_ = other.name_;
    signature_ = other.signature_;
  }
};

using SEN = SpecialEuclideanGroup;

#endif  // SPECIAL_EUCLIDEAN_GROUP_HPP