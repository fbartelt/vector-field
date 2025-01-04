#ifndef VECTORFIELD_HPP
#define VECTORFIELD_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <tuple>

#include "SpecialEuclideanGroup.h"

// Global constants for the default delta and ds values
float c_delta = 0.001;
float c_ds = 0.001;
// Approximate value for theta=0 in SE(3) EE distance
float c_maxCosTheta = 0.999;

template <typename Group>
class VectorField {
 protected:
  std::vector<Eigen::MatrixXd> curve;
  std::vector<Eigen::MatrixXd> curve_derivative;
  float ds_;
  float delta_;

  Eigen::VectorXd tangentComponent(float min_dist, int min_index) {
    Eigen::MatrixXd closest_point = curve.at(min_index);

    Eigen::MatrixXd dHd;
    if (curve_derivative.size() > 0) {
      dhd = curve_derivative.at(min_index);
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
    std::tuple<float, int>[min_distance, closest_index] = ECdistance(state);
    return tangentComponent(min_distance, closest_index);
  }

  Eigen::VectorXd normalComponent(const Group& state, float min_dist,
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
      float dDistance = EEdistance(variation, closest_point);
      LvDhat(i) = (dDistance - min_distance) / (delta_);
    }
    normal = -LvDhat;
    // normal += (gradD * state.algebra_.SR(state.matrix().col(i),
    // state.n())).transpose().eval();
    return normal;
  }

  Eigen::VectorXd normalComponent(const Group& state) {
    std::tuple<float, int>[min_distance, closest_index] = ECdistance(state);
    return normalComponent(state, min_distance, closest_index);
  }

  float kn(float distance, float a1 = 1.0, float a2 = 1.0) {
    return a1 * std::tanh(a2 * distance);
  }

  float kt(float distance, float a1 = 1.0, float a2 = 1.0, float a3 = 1.0) {
    return a1 * (1 - a2 * std::tanh(a3 * distance));
  }

  void saveData(const Eigen::MatrixXd& nearest, const Eigen::VectorXd& tangent,
                const Eigen::VectorXd& normal, const float distance) {
    // Save the data of the iteration
    iterationResults.push_back(
        std::make_tuple(nearest, tangent, normal, distance));
  }

 public:
  std::vector<
      std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, float>>
      iterationResults;

  VectorField(const std::vector<Eigen::MatrixXd>& curve, float delta = c_delta,
              float ds_ = c_ds)
      : curve(curve), delta_(delta), ds_(ds_) {};

  VectorField(const std::vector<Eigen::MatrixXd>& curve,
              const std::vector<Eigen::MatrixXd>& curve_derivative,
              float delta = c_delta, float ds_ = c_ds)
      : curve(curve),
        curve_derivative(curve_derivative),
        delta_(delta),
        ds_(ds_) {};

  Eigen::VectorXd operator()(const Group& state) { return eval(state); };

  float EEdistance(const Group& state, const Eigen::MatrixXd& W) {
    Eigen::MatrixXd V = state.matrix();
    return ((V.inverse() * W).log()).norm();
  }

  std::tuple<float, int> ECdistance(const Group& state) {
    Eigen::MatrixXd V = state.matrix();
    int ind_min = 0;
    float min_distance = 1e6;
    for (int i = 0; i < curve.size(); i++) {
      float distance = EEdistance(state, curve.at(i));
      if (distance < min_distance) {
        min_distance = distance;
        ind_min = i;
      }
    }
    return {min_distance, ind_min};
  }

  Eigen::VectorXd eval(const Group& state, bool save_data = false,
                       float gain_n1 = 1.0, float gain_n2 = 1.0,
                       float gain_t1 = 1.0, float gain_t2 = 1.0,
                       float gain_t3 = 1.0) {
    auto [min_distance, closest_index] = ECdistance(state);
    Eigen::MatrixXd closest_point = curve.at(closest_index);

    Eigen::VectorXd normal =
        normalComponent(state, min_distance, closest_index);
    Eigen::VectorXd tangent = tangentComponent(min_distance, closest_index);

    tangent = kt(min_distance, gain_t1, gain_t2, gain_t3) * tangent;
    normal = kn(min_distance, gain_n1, gain_n2) * normal;
    if (save_data) {
      saveData(closest_point, tangent, normal, min_distance);
    }
    return tangent + normal;
  }

  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float,
             float, float>
  EEdistSE3Variables(const Eigen::MatrixXd& X) {
    // Compute the variables used in the explicit EEdistance function in SE(3)
    Eigen::MatrixXd Z = X;
    Eigen::Matrix3d Q = Z.block<3, 3>(0, 0);
    Eigen::Vector3d t = Z.block<3, 1>(0, 3);
    float cos_theta = 0.5 * (Q.trace() - 1);
    float sin_theta = 1 / (2 * sqrt(2)) * (Q - Q.inverse()).norm();
    float theta = atan2(sin_theta, cos_theta);
    cos_theta = cos(theta);
    sin_theta = sin(theta);

    float alpha;
    if (cos_theta > c_maxCosTheta) {
      alpha = -1 / 12;
    } else {
      alpha =
          (2.0 - 2 * cos_theta - pow(theta, 2)) / (4 * pow((1 - cos_theta), 2));
    }
    Eigen::Matrix3d M = alpha * (Q + Q.inverse()) +
                        (1 - 2 * alpha) * Eigen::Matrix3d::Identity();
    return std::make_tuple(Q, t, M, theta, cos_theta, sin_theta, alpha);
  }
  // Eigen::Matrix4d EEdistSE3_derivative(const Group& state,
  //                                      const Eigen::MatrixXd& p2,
  //                                      float distance);
  // Eigen::VectorXd lie_derivative(const Group& state,
  //                                const Eigen::MatrixXd& closest_point,
  //                                const float distance);
  // std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float,
  //            float, float>
  // EEdistSE3_variables(const Eigen::MatrixXd& arg);
};

template <>
inline float VectorField<SpecialEuclideanGroup>::EEdistance(
    const SpecialEuclideanGroup& state, const Eigen::MatrixXd& W) {
  Eigen::MatrixXd Z = state.matrix().inverse() * W;
  auto [Q, t, M, theta, cos_theta, sin_theta, alpha] = EEdistSE3Variables(Z);

  float distance = sqrt(2 * pow(theta, 2) + t.transpose() * M * t);
  return distance;
}

template <>
inline Eigen::VectorXd VectorField<SpecialEuclideanGroup>::normalComponent(
    const SpecialEuclideanGroup& state, float min_dist, int min_index) {
  // TODO
  // Throw an error of NOT IMPLEMEMENTED
  throw std::logic_error("FUNCTION NOT IMPLEMENTED");
  Eigen::VectorXd normal_component =
      Eigen::VectorXd::Zero(state.matrix().rows());
  return normal_component;
}

template <>
inline Eigen::VectorXd VectorField<SpecialEuclideanGroup>::normalComponent(
    const SpecialEuclideanGroup& state) {
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
//   std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float,
//              float, float>
//   VectorField::EEdistSE3Variables(const Eigen::MatrixXd& X) {
//     // Compute the variables used in the explicit EEdistance function
//     Eigen::MatrixXd Z = X;
//     Eigen::Matrix3d Q = Z.block<3, 3>(0, 0);
//     Eigen::Vector3d t = Z.block<3, 1>(0, 3);
//     float cos_theta = 0.5 * (Q.trace() - 1);
//     float sin_theta = 1 / (2 * sqrt(2)) * (Q - Q.inverse()).norm();
//     float theta = atan2(sin_theta, cos_theta);
//     cos_theta = cos(theta);
//     sin_theta = sin(theta);

//     float alpha;
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
//   float EEdistance(const SpecialEuclideanGroup& state,
//                    const Eigen::MatrixXd& W) {
//     Eigen::MatrixXd Z = state.matrix().inverse() * W;
//     auto [Q, t, M, theta, cos_theta, sin_theta, alpha] =
//     EEdistSE3Variables(Z);

//     float distance = sqrt(2 * pow(theta, 2) + t.transpose() * M * t);
//     return distance;
//   }

//   Eigen::VectorXd normalComponent(const SpecialEuclideanGroup& state,
//                                   float min_dist, int min_index) {
//     // TODO
//   }
//   Eigen::VectorXd normalComponent(const SpecialEuclideanGroup& state) {
//     auto [min_dist, min_index] = this->ECdistance(state);
//     return normalComponent(state, min_dist, min_index);
//   }
// }

#endif  // VECTORFIELD_HPP