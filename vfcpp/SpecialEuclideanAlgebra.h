#ifndef SPECIAL_EUCLIDEAN_ALGEBRA_H
#define SPECIAL_EUCLIDEAN_ALGEBRA_H

#include <eigen3/Eigen/Dense>

class SpecialEuclideanAlgebra {
  private:
    int dim_; // manifold dimension
    int n_;  // matrix dimension

  public:
    // Constructor
    SpecialEuclideanAlgebra(int n) : n_(n + 1) {
        // Manifold dimension for se Lie algebra
        dim_ = n * (n - 1) / 2 + n;
    }

    // Dimension of the Lie algebra
    int dim() const;

    // Implementation of print method for se
    void print() const;

    // Implementation of SL method using Eigen
    Eigen::MatrixXd SL(const Eigen::VectorXd& xi, int n) const;

    // Inverse operator for SL
    Eigen::VectorXd invSL(const Eigen::MatrixXd& X, int n) const;

    // Placeholder for SR method
    Eigen::MatrixXd SR(const Eigen::VectorXd& hi, int n) const;

};

#endif // SPECIAL_EUCLIDEAN_ALGEBRA_H