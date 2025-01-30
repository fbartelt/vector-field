#ifndef PROPER_ORTHOCHRONOUS_LORENTZ_GROUP_HPP
#define PROPER_ORTHOCHRONOUS_LORENTZ_GROUP_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "GeneralLinearAlgebra.hpp"
#include "GeneralLinearGroup.hpp"
#include "ProperOrthochronousLorentzAlgebra.hpp"

class ProperOrthochronousLorentzGroup
    : public GeneralLinearGroupBase<ProperOrthochronousLorentzGroup,
                                    ProperOrthochronousLorentzAlgebra> {
 private:
  int p_ = 3;
  int q_ = 1;

 public:
  ProperOrthochronousLorentzGroup()
      : GeneralLinearGroupBase<ProperOrthochronousLorentzGroup,
                               ProperOrthochronousLorentzAlgebra>(
            ProperOrthochronousLorentzAlgebra()) {
    n_ = 4;
    dim_ = 6;
    name_ = "SO⁺";
    signature_ = "(3,1)";
    matrix_ = random();
  }

  ProperOrthochronousLorentzGroup(const Eigen::MatrixXd& mat)
      : GeneralLinearGroupBase<ProperOrthochronousLorentzGroup,
                               ProperOrthochronousLorentzAlgebra>(
            ProperOrthochronousLorentzAlgebra()) {
    if (mat.rows() != 4 || mat.cols() != 4) {
      throw std::invalid_argument("Matrix dimensions must be 4x4.");
    }
    n_ = 4;
    dim_ = 6;
    name_ = "SO⁺";
    signature_ = "(3,1)";
    matrix_ = mat;
  }

  ProperOrthochronousLorentzGroup(const ProperOrthochronousLorentzGroup& other)
      : GeneralLinearGroupBase<ProperOrthochronousLorentzGroup,
                               ProperOrthochronousLorentzAlgebra>(
            ProperOrthochronousLorentzAlgebra()) {
    n_ = other.n_;
    dim_ = other.dim_;
    matrix_ = other.matrix_;
    name_ = other.name_;
    signature_ = other.signature_;
  }
};

using SO31 = ProperOrthochronousLorentzGroup;

#endif  // PROPER_ORTHOCHRONOUS_LORENTZ_GROUP_HPP