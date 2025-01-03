#ifndef LIEGROUP_HPP
#define LIEGROUP_HPP

#include <eigen3/Eigen/Dense>

template <typename AlgebraType>
class LieGroup {
protected:
    AlgebraType algebra_;
public:
    LieGroup(const AlgebraType& algebra = AlgebraType()) : algebra_(algebra) {}
    LieGroup(const LieGroup& other) : algebra_(other.algebra_) {}
    virtual ~LieGroup() = default;

    // Pure virtual method that must be implemented in derived classes
    virtual Eigen::MatrixXd random() const = 0;

    const AlgebraType& algebra() const {
        return algebra_;
    }
};

#endif // LIEGROUP_HPP