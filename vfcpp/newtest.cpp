#include "SpecialEuclideanGroup.hpp"
#include "SpecialEuclideanAlgebra.hpp"
#include "ProperOrthochronousLorentzGroup.hpp"
#include "ProperOrthochronousLorentzAlgebra.hpp"
#include "VectorField.hpp"

#include <eigen3/Eigen/Dense>
#include <iostream>

int main(){
  // SE3 X = SE3();
  // X.print();
  // Eigen::MatrixXd mat(4, 4);
  // mat << 0.5, 0.5, 0, 1,
  //        -0.5, 0.5, 0, 2,
  //        0, 0, 1, 3,
  //        0, 0, 0, 1;
  // Eigen::VectorXd xi(6);
  // xi << 1, 2, 3, 4, 5, 6;
  // Eigen::MatrixXd H(4, 4);
  // H << 0, -3, 2, 4,
  //      3, 0, -1, 8,
  //      -2, 1, 0, 12,
  //      0, 0, 0, 0;
  // se3 A2 = se3(H);
  // std::cout << "print A2" << std::endl;
  // A2.print();
  // se3 A = X.algebra().S(xi);
  // std::cout << "print A" << std::endl;
  // A.print();
  // std::cout << "print xi" << std::endl;
  // // Eigen::Matrix3d debug = Eigen::Matrix3d::Identity();
  // Eigen::VectorXd xi2 = X.algebra().invS(A);
  // std::cout << "invS(A) = " << xi2 << std::endl;
  // std::cout << "here" << std::endl;


  SO31 X = SO31();
  X.print();
  
  Eigen::VectorXd xi(6);
  xi << 1, 2, 3, 4, 5, 6;
  so31 A = X.algebra().S(xi);
  A.print();
  Eigen::VectorXd xi2 = X.algebra().invS(A);
  std::cout << "invS(A) = " << xi2 << std::endl;
  return 0;
}