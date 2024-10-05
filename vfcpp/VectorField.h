#ifndef VECTORFIELD_H
#define VECTORFIELD_H

#include <eigen3/Eigen/Dense>
#include "SpecialEuclideanGroup.h"
#include <tuple>

class VectorField{
  private:
    std::vector<Eigen::MatrixXd> curve;
    int divide_and_conquer(const SpecialEuclideanGroup& state, int curve_start, int curve_end);
    Eigen::VectorXd tangent_component(const SpecialEuclideanGroup& state);
    Eigen::VectorXd normal_component(const SpecialEuclideanGroup& state);
    Eigen::MatrixXd delta_matrix(float delta, int n, int row, int col);    
    float kn(float distance, float gain);
    float kt(float distance, float gain);
    void write_data(const Eigen::MatrixXd& nearest, const Eigen::VectorXd& tangent, const Eigen::VectorXd& normal, const float distance);
    float ds;
    float delta_;

  public:
    VectorField(const std::vector<Eigen::MatrixXd>& curve, float delta=0.001) : curve(curve), delta_(delta) { ds=0.001; }; //ds = 1.0 / curve.size(); 
    Eigen::VectorXd eval(const SpecialEuclideanGroup& state, bool save_data=false, float gain_n=1.0, float gain_t=1.0);
    Eigen::VectorXd operator()(const SpecialEuclideanGroup& state) { return eval(state); };
    float ECdistance(const SpecialEuclideanGroup& state);
    float EEdistance(const SpecialEuclideanGroup& state, const Eigen::MatrixXd& p2);
    std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, float>> iterationResults;
};

#endif // VECTORFIELD_H