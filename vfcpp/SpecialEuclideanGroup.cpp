#include "SpecialEuclideanGroup.h"
#include <eigen3/unsupported/Eigen/MatrixFunctions>


Eigen::MatrixXd SpecialEuclideanGroup::random(int n) const {
    // Step 1: Get the dimension of the associated Lie algebra
    int dim = dim_;  // Access the associated Lie algebra

    // Step 2: Create a dim-dimensional random vector XI
    Eigen::VectorXd XI = Eigen::VectorXd::Random(dim);

    // Step 3: Compute SL(XI, n) using the associated Lie algebra's SL function
    Eigen::MatrixXd SL_XI = algebra_.SL(XI, n);

    // Step 4: Compute and return the matrix exponential of SL(XI, n)
    Eigen::MatrixXd exp_SL_XI = SL_XI.exp(); // Matrix exponential
    return exp_SL_XI;
}

SpecialEuclideanGroup SpecialEuclideanGroup::operator+(const Eigen::MatrixXd& other) const {
    // Check if the dimensions match (optional)
    if (matrix_.rows() != other.rows() || matrix_.cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    // Perform the matrix addition
    Eigen::MatrixXd result = matrix_ + other;

    // Return a new SpecialEuclideanGroup object with the resulting matrix
    return SpecialEuclideanGroup(n_ - 1, result);
}

SpecialEuclideanGroup& SpecialEuclideanGroup::operator=(const SpecialEuclideanGroup& other) {
    if (this != &other) {
        n_ = other.n_;
        matrix_ = other.matrix_;
        dim_ = other.dim_;
    }
    return *this;
}

// Global function to overload the + operator for Eigen::MatrixXd + SpecialEuclideanGroup
SpecialEuclideanGroup operator+(const Eigen::MatrixXd& mat, const SpecialEuclideanGroup& element) {
    // Check if the dimensions match (optional)
    if (mat.rows() != element.matrix_.rows() || mat.cols() != element.matrix_.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    // Perform the matrix addition
    Eigen::MatrixXd result = mat + element.matrix_;
    // Changes the last row to enforce SE(n) constraints
    result.row(element.matrix().rows() - 1) << 0, 0, 0, 1;

    // Return a new SpecialEuclideanGroup object with the resulting matrix
    return SpecialEuclideanGroup(element.n(), result);
}
