#pragma once

#include <eigen3/Eigen/Dense>

template <typename DerivedAlgebra>
class LieAlgebra {
public:
    virtual ~LieAlgebra() = default;

    // Pure virtual methods that must be implemented in derived classes
    virtual Eigen::MatrixXd exp() const = 0;
    virtual DerivedAlgebra S(const Eigen::VectorXd& xi) const = 0;
    virtual Eigen::VectorXd invS(const Eigen::MatrixXd& matrix) const = 0;

    // Optional: You can add common functionality for all LieAlgebra classes here
};