#include "VectorField.h"

#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <tuple>

float MAX_VAL = 0.0;

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float, float, float> VectorField::EEdistSE3_variables(const Eigen::MatrixXd& arg) {
  // Compute the variables used in the explicit EEdistance function
  Eigen::MatrixXd p1 = arg;
  Eigen::Matrix3d Q = p1.block<3, 3>(0, 0);
  Eigen::Vector3d t = p1.block<3, 1>(0, 3);
  float cos_theta = 0.5 * (Q.trace() - 1);
  float sin_theta = 1/(2 * sqrt(2)) * (Q - Q.inverse()).norm();
  float theta = atan2(sin_theta, cos_theta);
  cos_theta = cos(theta);
  sin_theta = sin(theta);

  float alpha;
  if (cos_theta > 0.999) {
    alpha = -1/12;
  }
  else{
    alpha = (2.0 - 2*cos_theta - pow(theta, 2))/ (4*pow((1 - cos_theta), 2));
  }
  Eigen::Matrix3d M = alpha * (Q + Q.inverse()) + (1 - 2*alpha) * Eigen::Matrix3d::Identity();
  return std::make_tuple(Q, t, M, theta, cos_theta, sin_theta, alpha);
}

float VectorField::EEdistance(const SpecialEuclideanGroup& state,
                              const Eigen::MatrixXd& p2) {
  // Compute the Euclidean distance between two points
  Eigen::MatrixXd p1 = state.matrix();
  float distance;
  float alpha;
  if (p1.rows() == 4) {
    Eigen::MatrixXd H = p1.inverse() * p2;
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float, float, float> variables = EEdistSE3_variables(H);
    Eigen::Matrix3d Q = std::get<0>(variables);
    Eigen::Vector3d t = std::get<1>(variables);
    Eigen::Matrix3d M = std::get<2>(variables);
    float theta = std::get<3>(variables);
    float cos_theta = std::get<4>(variables);
    float sin_theta = std::get<5>(variables);
    alpha = std::get<6>(variables);
    // Compute the distance
    distance = sqrt(2*pow(theta, 2) + t.transpose().eval() * M * t);
    
    // std::cout << "EE-dist H: " << H << std::endl;
    double val = abs(distance - (H.log()).norm());
    float val2 = abs((Q * Q.transpose().eval() - Eigen::Matrix3d::Identity()).norm());
    // if (val > MAX_VAL && val2 < 1e-3) {
    //   std::cout << "HH:" << H << std::endl;
    //   std::cout << "distance: " << distance << std::endl;
    //   std::cout << "diff.: " << val << std::endl;
    //   MAX_VAL = val;
    //   std::cout << "logdist: " << H.log().norm() << std::endl;
    //   // Pauses program
    //   // std::cin.get();
    // }

    // std::cout << "distance: " << MAX_VAL << std::endl;
    // If distance is nan, print every variable
    if (std::isnan(distance)) {
      // Debugging alpha that seems to be overflowing
      std::cout << "alpha: " << alpha << std::endl;
      std::cout << "numerator: " << 2.0 - 2*cos_theta - pow(theta, 2) << std::endl;
      std::cout << "denominator: " << 4*pow((1 - cos_theta), 2) << std::endl;
      // std::cout << "Q: " << std::endl << Q << std::endl;
      // std::cout << "t: " << std::endl << t << std::endl;
      std::cout << "cos_theta: " << cos_theta << std::endl;
      std::cout << "sin_theta: " << sin_theta << std::endl;
      std::cout << "theta: " << theta << std::endl;
      // std::cout << "alpha: " << alpha << std::endl;
      // std::cout << "M: " << std::endl << M << std::endl;
      // std::cout << "t^T M t: " << t.transpose().eval() * M * t << std::endl;
      // Throw an exception to stop the program
      throw std::runtime_error("Distance is nan");
    }
  } else {
    distance = ((p1.inverse() * p2).log()).norm();
  }
  return distance;
}

float VectorField::ECdistance(const SpecialEuclideanGroup& state) {
  // Computes the minimum distance between state and the curve by finding the
  // closest point on the curve and getting the minimum EEdistance through
  // divide and conquer algorithm.

  // Find the closest point on the curve
  int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
  Eigen::MatrixXd closest_point = curve.at(closest_index);
  float min_distance = EEdistance(state, closest_point);

  return min_distance;
}

Eigen::Matrix4d VectorField::EEdistSE3_derivative(const SpecialEuclideanGroup& state, const Eigen::MatrixXd& p2, float distance){
  // TODO: this is giving absurd results for the orientation part. Smth in order 1e7. This happens with not so small distances (1.1e-3).
  Eigen::Matrix4d delDdelZ = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d Z = p2.inverse() * state.matrix();
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, float, float, float, float> variables = EEdistSE3_variables(Z);
  Eigen::Matrix3d Q = std::get<0>(variables);
  Eigen::Vector3d t = std::get<1>(variables);
  Eigen::Matrix3d M = std::get<2>(variables);
  float theta = std::get<3>(variables);
  float cos_theta = std::get<4>(variables);
  float sin_theta = std::get<5>(variables);
  float alpha = std::get<6>(variables);
  // Compute the derivative of the distance
  // If theta is almost zero, change to limit value
  float dalphadtheta = (theta + (1 - pow(theta, 2))*sin_theta - (theta + sin_theta)*cos_theta) / (2*pow(cos_theta - 1, 3));
  if (theta < 1e-3) {
    theta = 0;
    cos_theta = 1;
    sin_theta = 0;
    alpha = -1/12;
    dalphadtheta = 0;
  }
  // std::cout << "cos_theta: " << cos_theta << std::endl;
  // std::cout << "sin_theta: " << sin_theta << std::endl;
  // std::cout << "theta: " << theta << std::endl;
  // std::cout << "dalphadtheta: " << dalphadtheta << std::endl;
  Eigen::Matrix3d N = (Q + Q.inverse()) / 2 - Eigen::Matrix3d::Identity();
  Eigen::Matrix3d delthetadelQ = (cos_theta/(4*sin_theta + 1e-6) * (Q.inverse() - Q)) - sin_theta/2 * Eigen::Matrix3d::Identity();
  // std::cout << "delthetadelQ: " << std::endl << delthetadelQ << std::endl;
  Eigen::Matrix3d delDdelQ = (1 / distance) * ((2 * theta + dalphadtheta * t.transpose().eval() * N * t) * delthetadelQ + alpha * t * t.transpose().eval());
  // std::cout << "delDdelQ: " << std::endl << delDdelQ << std::endl;
  Eigen::Vector3d delDdelt = (1 / distance) * (t.transpose().eval() * M);
  // std::cout << "delDdelt: " << std::endl << delDdelt << std::endl;
  delDdelZ.block<3, 3>(0, 0) = delDdelQ;
  //allocates the vector delDdelt as a row vector to the last row
  delDdelZ.block<1, 3>(3, 0) = delDdelt.transpose().eval();
  return delDdelZ;
}

Eigen::VectorXd VectorField::tangent_component(const SpecialEuclideanGroup& state) {
  // Compute the tangent component of the vector field at the given state
  int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
  Eigen::MatrixXd closest_point = curve.at(closest_index);
  float min_distance = EEdistance(state, closest_point);
  std::cout << "min_distance: " << min_distance << std::endl;
  Eigen::MatrixXd dHd;
  if (closest_index == curve.size() - 1) {
    // If the closest point is the last point on the curve, the next point is
    // the first point (closed curve).
    dHd = (curve.at(0) - closest_point) / ds;
  }
  else {
    Eigen::MatrixXd next_point = curve.at(closest_index + 1);
    dHd = (next_point - closest_point) / ds;
  }
  // Corrects last element of dHd (it will always be one) since the subtraction disregards the group.
  dHd(3, 3) = 1;
  Eigen::VectorXd tangent = state.algebra_.invSL(dHd * closest_point.inverse(), state.n());
  return kt(min_distance, 1) * tangent;
}

Eigen::VectorXd VectorField::normal_component(const SpecialEuclideanGroup& state) {
  // Compute the normal component of the vector field at the given state
  int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
  Eigen::MatrixXd closest_point = curve.at(closest_index);
  float min_distance = EEdistance(state, closest_point);
  Eigen::VectorXd normal = Eigen::VectorXd::Zero(state.dim());

  for (int j = 0; j < state.matrix().rows(); j++) {
    // Compute the gradient of the distance function gradD, that is a row vector. Represented as a matrix
    Eigen::MatrixXd gradD = Eigen::MatrixXd::Zero(1, state.matrix().rows());
    for (int i = 0; i < state.matrix().rows() - 1; i++) {
      Eigen::MatrixXd delta = delta_matrix(delta_, state.matrix().rows(), i, j);
      float dDistance = EEdistance(state + delta, closest_point);
      gradD(0, i) = (dDistance - min_distance) / (delta_);
    }
    Eigen::MatrixXd aa = state.algebra_.SR(state.matrix().col(j), state.n());
    Eigen::MatrixXd aux = (gradD * aa);
    // The normal component is the tranpose of the row-vector aux
    normal += aux.transpose().eval();
    // normal += (gradD * state.algebra_.SR(state.matrix().col(i), state.n())).transpose().eval();
  }
  return -kn(min_distance, 1) * normal;
}

float VectorField::kn(float distance, float gain) {
  // Compute the normal gain
  return std::tanh(gain * distance);
}

float VectorField::kt(float distance, float gain) {
  // Compute the tangent gain
  return (1 - 0.5*std::tanh(gain * distance));
}

Eigen::MatrixXd VectorField::delta_matrix(float delta, int n, int row, int col) {
  // Creates a (n x n) matrix with delta in the col-th column and row-th row
  Eigen::MatrixXd delta_matrix = Eigen::MatrixXd::Zero(n, n);
  delta_matrix(row, col) = delta;
  return delta_matrix;
}

Eigen::VectorXd VectorField::lie_derivative(const SpecialEuclideanGroup& state, const Eigen::MatrixXd& closest_point, const float distance) {
  // Computes the Lie derivative of the distance function
  Eigen::VectorXd derivative = Eigen::VectorXd::Zero(state.dim());
  Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(state.dim(), state.dim());
  for (int i = 0; i < state.dim(); i++) {
    Eigen::VectorXd ei = basis.col(i);
    std::cout << "ei: " << ei << std::endl;
    SpecialEuclideanGroup state_plus = (state.algebra_.SL(ei, state.n()) * delta_).exp() * state;
    std::cout << "state_plus: " << std::endl << state_plus.matrix() << std::endl;
    derivative(i) = (EEdistance(state_plus, closest_point) - distance) / delta_;
    std::cout << "derivative" << derivative << std::endl;
  }
  return derivative;
}

int VectorField::divide_and_conquer(const SpecialEuclideanGroup& state, int curve_start,
                                    int curve_end) {
  // Divide and conquer algorithm to find the minimum distance. Returns the
  // corresponding index on the curve of the closest point.
  if (curve_start == curve_end) {
    // Base case: two points
    return curve_start;
  } else {
    // Recursive case: divide the curve into two parts
    int mid = (curve_start + curve_end) / 2;
    int left_index = divide_and_conquer(state, curve_start, mid);
    int right_index = divide_and_conquer(state, mid + 1, curve_end);
    float left_distance = EEdistance(state, curve.at(left_index));
    float right_distance = EEdistance(state, curve.at(right_index));

    // Return the index of the closest point
    return left_distance < right_distance ? left_index : right_index;
  }
}

void VectorField::write_data(const Eigen::MatrixXd& nearest, const Eigen::VectorXd& tangent,
                            const Eigen::VectorXd& normal, const float distance) {
  // Save the data of the iteration
  iterationResults.push_back(std::make_tuple(nearest, tangent, normal, distance));
}

// Each use divide_and_conquer to obtain the same thing. (Inneficient).
Eigen::VectorXd VectorField::eval(const SpecialEuclideanGroup& state, bool save_data, float gain_n, float gain_t) {
  // Evaluate the vector field at the given state
  // Compute normal component
  // Computes time taken to find the closest point
  // auto start = std::chrono::high_resolution_clock::now();
  int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
  // auto finish = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = finish - start;
  Eigen::MatrixXd closest_point = curve.at(closest_index);
  float min_distance = EEdistance(state, closest_point);
  // std::cout << "min_distance: " << min_distance << std::endl;
  Eigen::VectorXd normal = Eigen::VectorXd::Zero(state.dim());
  // Analytical derivative for comparison
  Eigen::MatrixXd delEdelZ = EEdistSE3_derivative(state, closest_point, min_distance);
  Eigen::MatrixXd analytical = delEdelZ * closest_point.inverse();
  // std::cout << "analytical matrix: " << std::endl << delEdelZ << std::endl;
  // std::cout << "analytical matrix post: " << std::endl << analytical << std::endl;
  
  for (int j = 0; j < state.matrix().rows(); j++) {
    // Compute the gradient of the distance function gradD, that is a row vector. Represented as a matrix
    // Symmetric derivative is used.
    Eigen::MatrixXd gradD = Eigen::MatrixXd::Zero(1, state.matrix().rows());
    // gradD = lie_derivative(state, closest_point, min_distance);
    
    for (int i = 0; i < state.matrix().rows() - 1; i++) {
      Eigen::MatrixXd delta = delta_matrix(delta_, state.matrix().rows(), i, j);
      float rDistance = EEdistance(state + delta, closest_point);
      float r2Distance = EEdistance(state + 2*delta, closest_point);
      float lDistance = EEdistance(state - delta, closest_point);
      float l2Distance = EEdistance(state - 2*delta, closest_point);
      gradD(0, i) = (-r2Distance + 8*rDistance - 8*lDistance + l2Distance) / (12*delta_);
    }
    // Prints the norm between the analytical and numerical derivative. gradD represents a row vector. Analytical is a 
    // matrix, each column corresponds to the respective j-th gradD.
    // std::cout << "Comparison of derivatives: " << std::endl;
    Eigen::MatrixXd analytical_jthcol = analytical.row(j);
    float norm_diff = (gradD - analytical_jthcol).norm();
    // std::cout << ".a. " << (gradD - analytical_jthcol) << "Norm: " << norm_diff << std::endl;
    // if (norm_diff > 1e-3){
    //   std::cout << "row: " << j << std::endl;
    //   std::cout << "dist: " << min_distance << std::endl;
    //   std::cout << "Analytical: " << std::endl << analytical_jthcol << std::endl;
    //   std::cout << "Numerical: " << std::endl << gradD << std::endl;
    //   // std::cin.get();
    // }
    // gradD = analytical.row(j);
    Eigen::MatrixXd aa = state.algebra_.SR(state.matrix().col(j), state.n());
    Eigen::MatrixXd aux = (gradD * aa);
    // The normal component is the tranpose of the row-vector aux
    normal += aux.transpose().eval();
  }
  // Eigen::VectorXd analytical_normal = lie_derivative(state, closest_point, min_distance);
  // std::cout << "Analytical normal: " << std::endl << analytical_normal << std::endl;
  // std::cout << "Numerical normal: " << std::endl << normal << std::endl;
  // // normal = analytical_normal;
  // // Norm of difference
  // std::cout << "Norm of difference: " << (analytical_normal - normal).norm() << std::endl;
  // std::cin.get();

  // Compute tangent component
  Eigen::MatrixXd dHd;
  if (closest_index == curve.size() - 1) {
    // If the closest point is the last point on the curve, the next point is
    // the first point (closed curve).
    dHd = (curve.at(0) - closest_point) / ds;
  }
  else {
    Eigen::MatrixXd next_point = curve.at(closest_index + 1);
    dHd = (next_point - closest_point) / ds;
  }
  // Corrects last element of dHd (it will always be one) since the subtraction disregards the group.
  dHd(3, 3) = 1;
  Eigen::VectorXd tangent = state.algebra_.invSL(dHd * closest_point.inverse(), state.n());
  // Eigen::VectorXd tangent = tangent_component(state);
  // Eigen::VectorXd normal = normal_component(state);
  tangent = kt(min_distance, gain_t) * tangent;
  normal = - kn(min_distance, gain_n) * normal;
  if (save_data) {
    // Provisorily store the time to compute nearest point at the last element of the tuple
    // the time to compute is stored in seconds
    // write_data(closest_point, tangent, normal, elapsed.count());
    write_data(closest_point, tangent, normal, min_distance);
  }
  return tangent + normal;
}