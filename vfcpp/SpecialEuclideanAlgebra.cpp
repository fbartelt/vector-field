#include "SpecialEuclideanAlgebra.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

int SpecialEuclideanAlgebra::dim() const {
    return dim_;
}

void SpecialEuclideanAlgebra::print() const {
    std::cout << "This is the se(" << n_ << ") Lie algebra." << std::endl;
}

Eigen::MatrixXd SpecialEuclideanAlgebra::SL(const Eigen::VectorXd& xi, int n) const {
    // Check if the dimension of xi matches the Lie algebra dimension
    if (xi.size() != dim_) {
        throw std::invalid_argument("xi has the wrong dimension.");
    }

    // Create an nxn matrix filled with zeros using Eigen
    Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(n_, n_);

    // Get the indices for the upper triangular part of the matrix (excluding the diagonal)
    // Portion of Lie algebra related to SO(n)
    int index = dim_ - 1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            S_(i, j) = pow((-1), index) * xi(index);
            index--;
        }
    }

    S_ = S_ - S_.transpose().eval();

    // Portion of Lie algebra related to R^n
    for (int i = 0; i < n; ++i) {
        S_(i, n) = xi(i);
    }

    return S_;
}

Eigen::VectorXd SpecialEuclideanAlgebra::invSL(const Eigen::MatrixXd& X, int n) const {
    // Check if the dimension of X matches the Lie algebra dimension
    if (X.rows() != n_ || X.cols() != n_) {
        throw std::invalid_argument("X has the wrong dimension.");
    }

    // Create a vector of size dim_ filled with zeros using Eigen
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(dim_);

    // Get the indices for the upper triangular part of the matrix (excluding the diagonal)
    // Portion of Lie algebra related to SO(n)
    int index = dim_ - 1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Emulates the basis for SE(3) -- upper triangular as -z y; -x. So every odd xi_i is negated and the matrix is
            // populated in reverse order
            xi(index) = pow((-1), index) * X(i, j);
            index--;
        }
    }

    // Portion of Lie algebra related to R^n
    for (int i = 0; i < n; ++i) {
        xi(i) = X(i, n);
    }

    return xi;
}

Eigen::MatrixXd SpecialEuclideanAlgebra::SR(const Eigen::VectorXd& hi, int n) const {
    // Create an identity matrix of size dim x dim
    int dim = dim_;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);

    // Initialize the final matrix S_ which will hold the results
    Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(n_, dim);

    // Loop over each column of the identity matrix
    for (int i = 0; i < dim; ++i) {
        // Extract the i-th column of the identity matrix
        Eigen::VectorXd e = I.col(i);

        // Apply SL(e, n) to get an nxn matrix
        Eigen::MatrixXd SL_e = SL(e, n);
        // Assign the result to the i-th column of S_
        S_.col(i) = SL_e * hi;
    }

    return S_;
}