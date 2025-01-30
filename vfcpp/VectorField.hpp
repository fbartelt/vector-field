#ifndef VECTORFIELD_HPP
#define VECTORFIELD_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <tuple>

#include "SpecialEuclideanGroup.hpp"

// Global constants for the default delta and ds values
double c_delta = 0.001;
double c_ds = 0.001;
// Approximate value for theta=0 in SE(3) EE distance
double c_maxCosTheta = 0.999;

template <typename Group>
class VectorField {
 protected:
  std::vector<Eigen::MatrixXd> curve;
  std::vector<Eigen::MatrixXd> curve_derivative;
  double ds_;
  double delta_;

  Eigen::VectorXd tangentComponent(const Group& state, double min_dist, int min_index) {
    Eigen::MatrixXd closest_point = curve.at(min_index);

    Eigen::MatrixXd dHd;
    if (curve_derivative.size() > 0) {
      dHd = curve_derivative.at(min_index);
    } else {
      if (min_index == curve.size() - 1) {
        // If the closest point is the last point on the curve, the next point
        // is the first point (closed curve).
        dHd = (curve.at(0) - closest_point) / ds_;
      } else {
        Eigen::MatrixXd next_point = curve.at(min_index + 1);
        dHd = (next_point - closest_point) / ds_;
      }
    }
    Eigen::VectorXd tangent =
        state.algebra().invS(dHd * closest_point.inverse());
    return tangent;
  }

  Eigen::VectorXd tangentComponent(const Group& state) {
    auto [min_distance, closest_index] = ECdistance(state);
    return tangentComponent(min_distance, closest_index);
  }

  Eigen::VectorXd normalComponent(const Group& state, double min_dist,
                                  int min_index) {
    // Compute the normal component of the vector field at the given state
    // int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
    Eigen::MatrixXd closest_point = curve.at(min_index);
    int m = state.dim();
    Eigen::VectorXd normal = Eigen::VectorXd::Zero(m);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd LvDhat =
        Eigen::VectorXd::Zero(m);  // L-operator wrt V of EEdistance

    for (int i = 0; i < m; i++) {
      Eigen::MatrixXd variation =
          (state.algebra().S(I.col(i)) * delta_).exp() * state;
      double dDistance = EEdistance(variation, closest_point);
      LvDhat(i) = (dDistance - min_dist) / (delta_);
    }
    normal = -LvDhat;
    // normal += (gradD * state.algebra_.SR(state.matrix().col(i),
    // state.n())).transpose().eval();
    return normal;
  }

  Eigen::VectorXd normalComponent(const Group& state) {
    auto [min_distance, closest_index] = ECdistance(state);
    return normalComponent(state, min_distance, closest_index);
  }

  double kn(double distance, double a1 = 1.0, double a2 = 1.0) {
    return a1 * std::tanh(a2 * distance);
  }

  double kt(double distance, double a1 = 1.0, double a2 = 1.0, double a3 = 1.0) {
    return a1 * (1 - a2 * std::tanh(a3 * distance));
  }

  void saveData(const Eigen::MatrixXd& nearest, const Eigen::VectorXd& tangent,
                const Eigen::VectorXd& normal, const double distance) {
    // Save the data of the iteration
    iterationResults.push_back(
        std::make_tuple(nearest, tangent, normal, distance));
  }

 public:
  std::vector<
      std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double>>
      iterationResults;
  std::vector<double> norm_hist;
  std::vector<double> closest_indices;

  VectorField(const std::vector<Eigen::MatrixXd>& curve, double delta = c_delta,
              double ds_ = c_ds)
      : curve(curve), delta_(delta), ds_(ds_) {};

  VectorField(const std::vector<Eigen::MatrixXd>& curve,
              const std::vector<Eigen::MatrixXd>& curve_derivative,
              double delta = c_delta, double ds_ = c_ds)
      : curve(curve),
        curve_derivative(curve_derivative),
        delta_(delta),
        ds_(ds_) {};

  Eigen::VectorXd operator()(const Group& state) { return eval(state); };

  int kronDelta(int i, int j) { return (i == j) ? 1 : 0; }

  double EEdistance(const Group& state, const Eigen::MatrixXd& W) {
    Eigen::MatrixXd V = state.matrix();
    return ((V.inverse() * W).log()).norm();
  }

  std::tuple<double, int> ECdistance(const Group& state) {
    Eigen::MatrixXd V = state.matrix();
    int ind_min = 0;
    double min_distance = 1e6;
    for (int i = 0; i < curve.size(); i++) {
      double distance = EEdistance(state, curve.at(i));
      if (distance < min_distance) {
        min_distance = distance;
        ind_min = i;
      }
    }
    return {min_distance, ind_min};
  }

  Eigen::VectorXd eval(const Group& state, bool save_data = false,
                       double gain_n1 = 1.0, double gain_n2 = 1.0,
                       double gain_t1 = 1.0, double gain_t2 = 1.0,
                       double gain_t3 = 1.0) {
    auto [min_distance, closest_index] = ECdistance(state);
    Eigen::MatrixXd closest_point = curve.at(closest_index);

    Eigen::VectorXd normal =
        normalComponent(state, min_distance, closest_index);
    Eigen::VectorXd tangent = tangentComponent(state, min_distance, closest_index);

    tangent = kt(min_distance, gain_t1, gain_t2, gain_t3) * tangent;
    normal = kn(min_distance, gain_n1, gain_n2) * normal;
    if (save_data) {
      saveData(closest_point, tangent, normal, min_distance);
      closest_indices.push_back(closest_index);
    }
    return tangent + normal;
  }

  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, double, double,
             double, double>
  EEdistSE3Variables(const Eigen::MatrixXd& X) {
    // Compute the variables used in the explicit EEdistance function in SE(3)
    Eigen::MatrixXd Z = X;
    Eigen::Matrix3d Q = Z.block<3, 3>(0, 0);
    Eigen::Vector3d t = Z.block<3, 1>(0, 3);
    double cos_theta = 0.5 * (Q.trace() - 1);
    double sin_theta = 1.0 / (2.0 * sqrt(2)) * (Q - Q.inverse()).norm();
    double theta = atan2(sin_theta, cos_theta);
    cos_theta = cos(theta);
    sin_theta = sin(theta);

    double alpha;
    if (cos_theta > c_maxCosTheta) {
      alpha = -(1.0 / 12);
    } else {
      alpha =
          (2.0 - 2 * cos_theta - pow(theta, 2)) / (4.0 * pow((1 - cos_theta), 2));
    }
    Eigen::Matrix3d M = alpha * (Q + Q.inverse()) +
                        (1 - 2 * alpha) * Eigen::Matrix3d::Identity();
    return std::make_tuple(Q, t, M, theta, cos_theta, sin_theta, alpha);
  }

  Eigen::Matrix3d Shat(const Eigen::VectorXd& omega){
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    A(0, 1) = -omega(2);
    A(0, 2) = omega(1);
    A(1, 2) = -omega(0);
    A = A - A.transpose().eval();
    return A;
  }
  // Eigen::Matrix4d EEdistSE3_derivative(const Group& state,
  //                                      const Eigen::MatrixXd& p2,
  //                                      double distance);
  // Eigen::VectorXd lie_derivative(const Group& state,
  //                                const Eigen::MatrixXd& closest_point,
  //                                const double distance);
  // std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, double, double,
  //            double, double>
  // EEdistSE3_variables(const Eigen::MatrixXd& arg);
};

template <>
inline double VectorField<SE3>::EEdistance(
    const SE3& state, const Eigen::MatrixXd& W) {
  Eigen::MatrixXd Z = state.matrix().inverse() * W;
  auto [Q, t, M, theta, cos_theta, sin_theta, alpha] = EEdistSE3Variables(Z);

  double distance = sqrt(2.0 * pow(theta, 2) + t.transpose() * M * t);
  return distance;
}

template <>
inline Eigen::VectorXd VectorField<SE3>::normalComponent(
    const SE3& state, double min_dist, int min_index) {
  // TODO
  // std::cout << "nromal comp" << std::endl;
  // Throw an error of NOT IMPLEMEMENTED
  Eigen::MatrixXd W = curve.at(min_index);
  Eigen::MatrixXd Z = W.matrix().inverse() * state.matrix();
  auto [Q, u, X_bar, theta, cos_theta, sin_theta, beta0] = EEdistSE3Variables(Z);
  Eigen::Matrix3d Q2 = Q * Q;
  double Q_trace = Q.trace();
  double Q2_trace = Q2.trace();
  Eigen::VectorXd g_vec = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd f_vec = Eigen::VectorXd::Zero(6);

  Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
  Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
  Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d Shat_e1_tranZ = Shat(ex) * u;
  Eigen::Vector3d Shat_e2_tranZ = Shat(ey) * u;
  Eigen::Vector3d Shat_e3_tranZ = Shat(ez) * u;
  
  Eigen::Matrix3d Shat_e1_Q = Shat(ex) * Q;
  Eigen::Matrix3d Shat_e2_Q = Shat(ey) * Q;
  Eigen::Matrix3d Shat_e3_Q = Shat(ez) * Q;

  // g_vec(3) = (Q(1, 2) - Q(2, 1));
  // g_vec(4) = (Q(2, 0) - Q(0, 2));
  // g_vec(5) = (Q(0, 1) - Q(1, 0));
  g_vec(3) = Shat_e1_Q.trace();
  g_vec(4) = Shat_e2_Q.trace();
  g_vec(5) = Shat_e3_Q.trace();
  // g_vec = g_vec/2;
  // std::cout << "Computed g_vec" << std::endl;
  // f_vec(3) = (Q2(1, 2) - Q2(2, 1));
  // f_vec(4) = (Q2(2, 0) - Q2(0, 2));
  // f_vec(5) = (Q2(0, 1) - Q2(1, 0));
  f_vec(3) = (Shat(ex) * Q2).trace();
  f_vec(4) = (Shat(ey) * Q2).trace();
  f_vec(5) = (Shat(ez) * Q2).trace();

  Eigen::VectorXd f_test = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd g_test = Eigen::VectorXd::Zero(6);
  f_test(3) = Q2(1, 2) - Q2(2, 1);
  f_test(4) = Q2(2, 0) - Q2(0, 2);
  f_test(5) = Q2(0, 1) - Q2(1, 0);
  g_test(3) = Q(1, 2) - Q(2, 1);
  g_test(4) = Q(2, 0) - Q(0, 2);
  g_test(5) = Q(0, 1) - Q(1, 0);

  // f_vec = f_vec/2;
  // std::cout << "Computed f_vec" << std::endl;
  // std::cout << "f_vec:" << f_vec << std::endl;
  // std::cout << "g_vec:" << g_vec << std::endl;

  // double Ltheta_den = (3/4) - ((1/4) * Q2_trace) + ((1/4) * pow((Q_trace - 1), 2));
  // std::cout << "Q_trace: " << Q_trace << std::endl;
  // std::cout << "Q2_trace: " << Q2_trace << std::endl;
  // std::cout << "Ltheta_den: " << Ltheta_den << std::endl;
  double sqrt_term = sqrt(3 - Q2_trace);
  // double g_mult = -sqrt_term / 2;
  // double f_mult = -((1/2) * (Q_trace - 1)) / (4 * sqrt_term);
  // double f_mult = -1 / (4 * sqrt(3 - Q2_trace)) * cos_theta;
  // double f_mult = -1/(8 * sin_theta) * cos_theta;
  double f_mult = -2.0*cos_theta / (4.0 * sqrt(3 - Q2_trace));
  // if (cos_theta > c_maxCosTheta){
  //   f_mult = 0;
  // }
  double g_mult = -sin_theta / 2.0;
  // std::cout << "g_mult: " << g_mult << std::endl;
  // std::cout << "f_mult: " << f_mult << std::endl;
  // Eigen::VectorXd Ltheta = (f_mult * f_vec + g_mult * g_vec) / (Ltheta_den + 1e-6);
  Eigen::VectorXd Ltheta = ((f_mult * f_vec) + (g_mult * g_vec));
  // std::cout << "Ltheta: " << Ltheta << std::endl;

  Eigen::MatrixXd Q_T = Q.transpose().eval();
  Eigen::Matrix3d Shat_e1_Q_T = Shat(ex) * Q_T;
  Eigen::Matrix3d Shat_e2_Q_T = Shat(ey) * Q_T;
  Eigen::Matrix3d Shat_e3_Q_T = Shat(ez) * Q_T;
  // std::cout << "Shat(e1): " << Shat_e1_Q << std::endl;
  // std::cout << "Shat(e2): " << Shat_e2_Q << std::endl;
  // std::cout << "Shat(e3): " << Shat_e3_Q << std::endl;

  Eigen::VectorXd LtranZi = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd Lpos = Eigen::VectorXd::Zero(6); // L[u^T X u]
  // double Lbeta_num = (pow(theta, 2) * sin_theta) - theta - sin_theta + ((theta + sin_theta) * cos_theta);
  // double Lbeta_den = 2 * pow(1 - cos_theta, 3);
  double Lbeta_num = (-pow(theta, 2) * sin_theta) + theta + sin_theta - ((theta + sin_theta) * cos_theta);
  double Lbeta_den = 2.0 * pow(cos_theta - 1, 3);
  Eigen::VectorXd Lbeta0 = (Lbeta_num / Lbeta_den) * Ltheta;
  Eigen::VectorXd L_Qij = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd L_Q_T_ij = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd L_Xij = Eigen::VectorXd::Zero(6);
  // std::cout << "Computing Lpos" << std::endl;

  for(int i=0; i<3; i++){
    double tranZi = u(i);
    // std::cout << "i: " << i << std::endl;
    LtranZi(0) = kronDelta(i, 0);
    LtranZi(1) = kronDelta(i, 1);
    LtranZi(2) = kronDelta(i, 2);
    LtranZi(3) = Shat_e1_tranZ(i);
    LtranZi(4) = Shat_e2_tranZ(i);
    LtranZi(5) = Shat_e3_tranZ(i);
    // std::cout << "LtranZ_" << i+1 << ": " << LtranZi.transpose().eval() << std::endl;

    for(int j=0; j<3; j++){
      // std::cout << "j: " << j << std::endl;
      double tranZj = u(j);
      double Xij = X_bar(i, j);
      
      L_Qij(3) = Shat_e1_Q(i, j);
      L_Qij(4) = Shat_e2_Q(i, j);
      L_Qij(5) = Shat_e3_Q(i, j);
      L_Q_T_ij(3) = Shat_e1_Q(j, i);
      L_Q_T_ij(4) = Shat_e2_Q(j, i);
      L_Q_T_ij(5) = Shat_e3_Q(j, i);
      // L_Q_T_ij(3) = Shat_e1_Q_T(i, j);
      // L_Q_T_ij(4) = Shat_e2_Q_T(i, j);
      // L_Q_T_ij(5) = Shat_e3_Q_T(i, j);

      L_Xij = (Lbeta0 * (-2*kronDelta(i, j) + Q(i,j) + Q(j,i))) + (beta0 * (L_Qij + L_Q_T_ij));
      // std::cout << "L[Qij]" << L_Qij << std::endl;
      // std::cout << "L[Q^T_ij]" << L_Xij << std::endl;
      Lpos += (2*LtranZi * tranZj * Xij) + (tranZi * tranZj * L_Xij);
    }
  }
  // std::cout << "Lpos: " << Lpos << std::endl;

  Eigen::VectorXd L_Ehat;
  L_Ehat = (1 / (2.0*min_dist + 1e-6)) * ((4.0 * theta * Ltheta) + Lpos);
  // std::cout << "Computed L[E]" << std::endl;

  Eigen::MatrixXd Z_chain = Eigen::MatrixXd::Zero(6, 6);
  Eigen::MatrixXd Rd = W.block<3, 3>(0, 0);
  Eigen::VectorXd pd = W.block<3, 1>(0, 3);
  Eigen::MatrixXd Rd_T = Rd.transpose().eval();
  
  Z_chain.block<3, 3>(0, 0) = Rd_T;
  Z_chain.block<3, 3>(0, 3) = -Rd_T * Shat(pd);
  Z_chain.block<3, 3>(3, 3) = Rd_T;

  // std::cout << "Z_chain:" << Z_chain << std::endl;
  // std::cout << "L[E]:" << L_Ehat << std::endl;  
  
  Eigen::VectorXd normal_component = -L_Ehat.transpose().eval() * Z_chain;
  // std::cout << "Return normal:" << normal_component << std::endl;
  

  // Eigen::MatrixXd closest_point = curve.at(min_index);
  int m = 6;
  Eigen::VectorXd normal = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
  Eigen::MatrixXd LvDhat = Eigen::VectorXd::Zero(m);  // L-operator wrt V of EEdistance
  Eigen::MatrixXd LvTheta = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LvPos = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LvBeta = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LvCos = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LvSin = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ11 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ12 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ13 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ21 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ22 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ23 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ31 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ32 = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd LQ33 = Eigen::VectorXd::Zero(m);



  double eps = 1e-5;

  for (int i = 0; i < m; i++) {
    Eigen::MatrixXd variation =
        (state.algebra().S(I.col(i)) * eps).exp() * state;
    double dDistance = EEdistance(variation, W);
    // Eigen::MatrixXd Z_ = (state.algebra().S(I.col(i)) * eps).exp() * Z;
    Eigen::MatrixXd Z_ = W.matrix().inverse() * variation.matrix();
    auto [Q_, u_, X_bar_, theta_, cos_theta_, sin_theta_, beta0_] = EEdistSE3Variables(Z_);
    LvDhat(i) = (dDistance - min_dist) / (eps);
    LvTheta(i) = (theta_ - theta) / eps;
    double p_cur = (u.transpose().eval() * Q * u);
    double p_next = (u_.transpose().eval() * Q_ * u_);
    LvPos(i) = (p_next - p_cur) / (eps);
    LvBeta(i) = (beta0_ - beta0) / eps;
    LvCos(i) = (cos_theta_ - cos_theta) / eps;
    LvSin(i) = (sin_theta_ - sin_theta) / eps;
    LQ11(i) = (Q_(0, 0) - Q(0, 0)) / eps;
    LQ12(i) = (Q_(0, 1) - Q(0, 1)) / eps;
    LQ13(i) = (Q_(0, 2) - Q(0, 2)) / eps;
    LQ21(i) = (Q_(1, 0) - Q(1, 0)) / eps;
    LQ22(i) = (Q_(1, 1) - Q(1, 1)) / eps;
    LQ23(i) = (Q_(1, 2) - Q(1, 2)) / eps;
    LQ31(i) = (Q_(2, 0) - Q(2, 0)) / eps;
    LQ32(i) = (Q_(2, 1) - Q(2, 1)) / eps;
    LQ33(i) = (Q_(2, 2) - Q(2, 2)) / eps;
  }
  normal = -LvDhat;

  // std::cout << "L[cos]: " << (g_vec/2.0).transpose().eval() << std::endl;
  // std::cout << "Lvcos: " << LvCos.transpose().eval() << std::endl;
  // std::cout << "L[sin]: " << (-2.0*f_vec/((4.0 * sqrt(3 - Q2_trace)))).transpose().eval() << std::endl;
  // std::cout << "Lvsin: " << LvSin.transpose().eval() << std::endl;
  // Compares normal and normal_component element-wise
  std::cout << "normal explicit" << normal_component.transpose().eval() << std::endl;
  std::cout << "normal approx" << normal.transpose().eval() << std::endl;
  // DEBUGGING
  // LtransZi -- CORRECT !!
  // L[cos] -- 3e-3 error ~ok
  // L[sin] -- 3e-3 error ~ok
  // L[theta] -- 3e-3 error ~ok
  // L[beta] -- Wrong ?? 30e0 error
  // Lpos -- 6e-3 error ~ok
  // L[Qij] -- CORRECT !!
  // L[X_ij] -- Didnt check, must be ok cuz Lpos
  // std::cout << "Ltheta: " << Ltheta.transpose().eval()  << std::endl;
  // std::cout << "Ltheta_approx: " << LvTheta.transpose().eval() << std::endl;
  std::cout << "Ltheta norm: " << ((Ltheta.transpose().eval() * Z_chain) - LvTheta.transpose().eval()).norm() << std::endl;
  // std::cout << "Lpos: " << Lpos.transpose().eval()  << std::endl;
  // std::cout << "Lpos_approx: " << LvPos.transpose().eval() << std::endl;
  std::cout << "Lpos norm: " << ((Lpos.transpose().eval() * Z_chain) - LvPos.transpose().eval()).norm() << std::endl;
  std::cout << "Lbeta0: " << (Lbeta0.transpose().eval() * Z_chain)  << std::endl;
  std::cout << "Lbeta0_approx: " << LvBeta.transpose().eval() << std::endl;
  std::cout << "Lbeta norm: " << ((Lbeta0.transpose().eval() * Z_chain) - LvBeta.transpose().eval()).norm() << std::endl;
  // std::cout << "Ltheta: " << Ltheta.transpose().eval() * Z_chain << std::endl;


  for (int i=0; i<normal.size(); i++){
    double norm_ = abs(normal(i) - normal_component(i));
    // std::cout << "Normal error: " << norm_ << std::endl;
    // check if norm_ is nan:
    // if (norm_ != norm_){
    //   std::cout << "NAN ERROR" << std::endl;
    //   std::cout << "Q2 trace: " << Q2_trace << std::endl;
    //   throw std::runtime_error("NAN ERROR");
    // }
  }
  norm_hist.push_back((normal - normal_component).norm());
  std::cout << "normal error tot: " << (normal - normal_component).norm() << std::endl;
  std::cout << "beta0: " << beta0 << std::endl;
  std::cout << "cos theta: " << cos_theta << std::endl;
  std::cout << "sin theta: " << sin_theta << std::endl;
  std::cout << "theta: " << theta << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  return normal_component;
}

template <>
inline Eigen::VectorXd VectorField<SE3>::normalComponent(
    const SE3& state) {
  auto [min_dist, min_index] = ECdistance(state);
  Eigen::VectorXd normal_component =
      normalComponent(state, min_dist, min_index);
  return normal_component;
}

// template <>
// class VectorField<SpecialEuclideanGroup>
//     : public VectorField<SpecialEuclideanGroup> {
//  public:
//  using VectorField<SpecialEuclideanGroup>::ECdistance;
//   std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, double, double,
//              double, double>
//   VectorField::EEdistSE3Variables(const Eigen::MatrixXd& X) {
//     // Compute the variables used in the explicit EEdistance function
//     Eigen::MatrixXd Z = X;
//     Eigen::Matrix3d Q = Z.block<3, 3>(0, 0);
//     Eigen::Vector3d t = Z.block<3, 1>(0, 3);
//     double cos_theta = 0.5 * (Q.trace() - 1);
//     double sin_theta = 1 / (2 * sqrt(2)) * (Q - Q.inverse()).norm();
//     double theta = atan2(sin_theta, cos_theta);
//     cos_theta = cos(theta);
//     sin_theta = sin(theta);

//     double alpha;
//     if (cos_theta > c_maxCosTheta) {
//       alpha = -1 / 12;
//     } else {
//       alpha =
//           (2.0 - 2 * cos_theta - pow(theta, 2)) / (4 * pow((1 - cos_theta),
//           2));
//     }
//     Eigen::Matrix3d M = alpha * (Q + Q.inverse()) +
//                         (1 - 2 * alpha) * Eigen::Matrix3d::Identity();
//     return std::make_tuple(Q, t, M, theta, cos_theta, sin_theta, alpha);
//   }
//   double EEdistance(const SpecialEuclideanGroup& state,
//                    const Eigen::MatrixXd& W) {
//     Eigen::MatrixXd Z = state.matrix().inverse() * W;
//     auto [Q, t, M, theta, cos_theta, sin_theta, alpha] =
//     EEdistSE3Variables(Z);

//     double distance = sqrt(2 * pow(theta, 2) + t.transpose() * M * t);
//     return distance;
//   }

//   Eigen::VectorXd normalComponent(const SpecialEuclideanGroup& state,
//                                   double min_dist, int min_index) {
//     // TODO
//   }
//   Eigen::VectorXd normalComponent(const SpecialEuclideanGroup& state) {
//     auto [min_dist, min_index] = this->ECdistance(state);
//     return normalComponent(state, min_dist, min_index);
//   }
// }

#endif  // VECTORFIELD_HPP