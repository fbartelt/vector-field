#include "GeneralLinearGroup.hpp"
#include "GeneralLinearAlgebra.hpp"
#include "ProperOrthochronousLorentzAlgebra.hpp"
#include "ProperOrthochronousLorentzGroup.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <string>

int main(){
    // Eigen::VectorXd xi(9);
    // xi << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    // std::cout << xi << std::endl;
    // GeneralLinearAlgebra gla(3);
    // GeneralLinearAlgebra A = gla.S(xi);
    // A.print();
    // Eigen::VectorXd v(3);
    // v << 1, 2, 3;
    // std::cout << "invS: " << std::endl << A.invS(A.matrix()) << std::endl;
    // std::cout << "invS2: " << std::endl << A.invS(A) << std::endl;
    // std::cout << "A.matrix(): " << std::endl << A.matrix() << std::endl;
    // std::cout << "A.matrix().exp: " << std::endl << A.matrix().exp() << std::endl;
    // Eigen::MatrixXd mat = A.exp();
    // std::cout <<"exp: " << std::endl << mat << std::endl;
    //  std::cout << "A * v" << std::endl << A * v << std::endl;
    // GeneralLinearAlgebra A2 = A * 2;
    // std::cout << "A * 2: " << std::endl << A2.matrix() << std::endl;


    // GeneralLinearGroup glg(3);
    // glg.print();
    // GeneralLinearGroup X(A.matrix());
    // X.print();
    // std::cout << "X rows: " << X.matrix().rows() << " X cols: " << X.matrix().cols() << std::endl;
    // std::cout << "X.n(): " << X.n() << std::endl;
    // std::cout << "X.dim(): " << X.dim() << std::endl;
    // // Test basic math operators
    // std::cout << "X + X: " << std::endl << X + X << std::endl;
    // std::cout << "X - X: " << std::endl << X - X << std::endl;
    // std::cout << "-X: " << std::endl << -X << std::endl;
    // GeneralLinearGroup Y = X * X;
    // Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
    // std::cout << "X * X: " << std::endl << Y.matrix() << std::endl;
    // std::cout << "X * I: " << std::endl << X * I << std::endl;
    // std::cout << "X * 2: " << std::endl << X * 2 << std::endl;
    // std::cout << "2 * X: " << std::endl << 2 * X << std::endl;
    // std::cout << "X / 2" << std::endl << X / 2 << std::endl;

    SO31 X = SO31();
    X.print();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(4, 4);
    std::cout << "X * I: " << std::endl << X * I << std::endl;
    Eigen::VectorXd xi(6);
    xi << 1, 2, 3, 4, 5, 6;
    so31 A = X.algebra().S(xi);
    A.print();
    std::cout << "invS: " << std::endl << A.invS(A.matrix()) << std::endl;
    Eigen::MatrixXd mat = A.exp();
    std::cout << "exp: " << std::endl << mat << std::endl;
    std::cout << "exp2: " << std::endl << A.exp() << std::endl;

    so31 B = so31();
    B.setMatrix(mat);

  return 0;
}