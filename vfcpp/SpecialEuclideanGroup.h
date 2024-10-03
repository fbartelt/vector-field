#ifndef SPECIAL_EUCLIDEAN_GROUP_H
#define SPECIAL_EUCLIDEAN_GROUP_H

#include "SpecialEuclideanAlgebra.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

class SpecialEuclideanGroup{
  private:
    int n_; // matrix dimension
    Eigen::MatrixXd matrix_;  // The matrix representation of SE(n)
    int dim_; // Dimension of the Lie algebra (n * (n - 1) / 2)
  public:
    const SpecialEuclideanAlgebra algebra_;  // Associated Lie algebra (by value)
    // Constructor passing only the integer n (random matrix initialization)
    SpecialEuclideanGroup(int n) : n_(n + 1), algebra_(SpecialEuclideanAlgebra(n)) {
        // Calculate dimension of the associated Lie algebra
        dim_ = n * (n - 1) / 2 + n;
        // Initialize matrix randomly (you can implement a proper random generator later)
        matrix_ = SpecialEuclideanGroup::random(n);
    }

    // Constructor with both n and a matrix
    SpecialEuclideanGroup(int n, const Eigen::MatrixXd& mat) : n_(n + 1), matrix_(mat), algebra_(SpecialEuclideanAlgebra(n)) {
        if (mat.rows() != n + 1 || mat.cols() != n + 1) {
            throw std::invalid_argument("Matrix dimensions must be nxn.");
        }

        // Calculate dimension of the associated Lie algebra
        dim_ = n * (n - 1) / 2 + n;
    }

    SpecialEuclideanGroup(const SpecialEuclideanGroup& other) 
        : n_(other.n_), matrix_(other.matrix_), dim_(other.dim_), algebra_(other.algebra_) {}

    SpecialEuclideanGroup& operator=(const SpecialEuclideanGroup& other);
    SpecialEuclideanGroup operator+(const Eigen::MatrixXd& other) const;
    // Friend function to overload the + operator for Eigen::MatrixXd + SpecialEuclideanGroup
    friend SpecialEuclideanGroup operator+(const Eigen::MatrixXd& mat, const SpecialEuclideanGroup& element);

    // Returns the dimension of the associated Lie algebra
    int dim() const {
        return dim_;
    }
    
    int n() const {
        return n_ - 1;
    }

    const Eigen::MatrixXd& matrix() const {
        return matrix_;
    }

    // Print the Lie group
    void print() const {
        std::cout << "This is the Lie group with matrix:" << std::endl << matrix_ << std::endl;
    }

    // Function to generate a random matrix in the Lie group
    Eigen::MatrixXd random(int n) const ;
};

#endif // SPECIAL_EUCLIDEAN_GROUP_H