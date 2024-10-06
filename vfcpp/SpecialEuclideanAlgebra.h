#ifndef SPECIAL_EUCLIDEAN_ALGEBRA_H
#define SPECIAL_EUCLIDEAN_ALGEBRA_H

#include <eigen3/Eigen/Dense>

class SpecialEuclideanAlgebra {
  private:
    int dim_; // manifold dimension
    int n_;  // matrix dimension
    Eigen::MatrixXd matrix_;

  public:
    // Constructor
    SpecialEuclideanAlgebra(int n) : n_(n + 1) {
        // Manifold dimension for se Lie algebra
        dim_ = n * (n - 1) / 2 + n;
    }

    SpecialEuclideanAlgebra(const Eigen::MatrixXd& matrix) : matrix_(matrix) {
        n_ = matrix.rows();
        int n = n_ - 1;
        dim_ = n * (n - 1) / 2 + n;
    }

    // Dimension of the Lie algebra
    int dim() const;

    // Implementation of print method for se
    void print() const;

    // Implementation of SL method using Eigen
    SpecialEuclideanAlgebra SL(const Eigen::VectorXd& xi, int n) const;
    Eigen::VectorXd invSL(const Eigen::MatrixXd& X, int n) const;
    Eigen::VectorXd invSL(const SpecialEuclideanAlgebra& X, int n) const;
    Eigen::MatrixXd SR(const Eigen::VectorXd& hi, int n) const;

    Eigen::MatrixXd exp() const;

    SpecialEuclideanAlgebra operator*(const float scalar) const;
    Eigen::VectorXd operator*(const Eigen::VectorXd& v) const;
};

#endif // SPECIAL_EUCLIDEAN_ALGEBRA_H