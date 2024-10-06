#include "VectorField.h"

#include <eigen3/unsupported/Eigen/MatrixFunctions>

float MAX_VAL = 0.0;

float VectorField::EEdistance(const SpecialEuclideanGroup& state,
                              const Eigen::MatrixXd& p2) {
  // Compute the Euclidean distance between two points
  Eigen::MatrixXd p1 = state.matrix();
  float distance;
  double alpha;
  if (p1.rows() == 4) {
    Eigen::MatrixXd H = p1.inverse() * p2;
    // Extract 3x3 rotation block from H
    Eigen::Matrix3d Q = H.block<3, 3>(0, 0);
    // Extract translation vector from H
    Eigen::Vector3d t = H.block<3, 1>(0, 3);
    // Compute cos theta from Q
    double cos_theta = 0.5 * (Q.trace() - 1);
    // Compute sin theta from Q
    double sin_theta = 1/(2 * sqrt(2)) * (Q - Q.transpose().eval()).norm();
    // Compute theta with atan2
    double theta = atan2(sin_theta, cos_theta);
    // Compute alpha. If cos(theta) is almost 1, alpha is taken as the limit -1/12
    if (cos_theta > 0.99) {
      alpha = -1/12;
    }
    else{
      alpha = (2.0 - 2*cos_theta - pow(theta, 2))/ (4*pow((1 - cos_theta), 2));
    }

    // Compute M
    Eigen::Matrix3d M = alpha * (Q + Q.transpose().eval()) + (1 - 2*alpha) * Eigen::Matrix3d::Identity();
    // Compute the distance
    distance = sqrt(2*pow(theta, 2) + t.transpose().eval() * M * t);
    
    // std::cout << "EE-dist H: " << H << std::endl;
    double val = abs(distance - (H.log()).norm());
    if (val > MAX_VAL){
      std::cout << "HH:" << H << std::endl;
      std::cout << "distance: " << distance << std::endl;
      std::cout << "diff.: " << val << std::endl;
      MAX_VAL = val;
      std::cout << "logdist: " << H.log().norm() << std::endl;
    }

    // Checks if the eigenvalues of M are positive
    Eigen::EigenSolver<Eigen::Matrix3d> es(M);
    Eigen::Vector3cd eigenvalues = es.eigenvalues();
    for (int i = 0; i < 3; i++) {
      if (eigenvalues(i).real() < 0) {
        std::cout << "M: " << std::endl << M << std::endl;
        std::cout << "eigenvalues: " << std::endl << eigenvalues << std::endl;
        throw std::runtime_error("M has negative eigenvalues");
      }
    }
    
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
  int closest_index = divide_and_conquer(state, 0, curve.size() - 1);
  Eigen::MatrixXd closest_point = curve.at(closest_index);
  float min_distance = EEdistance(state, closest_point);
  // std::cout << "min_distance: " << min_distance << std::endl;
  Eigen::VectorXd normal = Eigen::VectorXd::Zero(state.dim());

  for (int j = 0; j < state.matrix().rows(); j++) {
    // Compute the gradient of the distance function gradD, that is a row vector. Represented as a matrix
    // Symmetric derivative is used.
    Eigen::MatrixXd gradD = Eigen::MatrixXd::Zero(1, state.matrix().rows());
    for (int i = 0; i < state.matrix().rows() - 1; i++) {
      Eigen::MatrixXd delta = delta_matrix(delta_, state.matrix().rows(), i, j);
      float rDistance = EEdistance(state + delta, closest_point);
      float lDistance = EEdistance(state - delta, closest_point);
      gradD(0, i) = (rDistance - lDistance) / (2*delta_);
    }
    Eigen::MatrixXd aa = state.algebra_.SR(state.matrix().col(j), state.n());
    Eigen::MatrixXd aux = (gradD * aa);
    // The normal component is the tranpose of the row-vector aux
    normal += aux.transpose().eval();
  }

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
    write_data(closest_point, tangent, normal, min_distance);
  }
  return tangent + normal;
}