#include "SpecialEuclideanGroup.hpp"
#include "SpecialEuclideanAlgebra.hpp"
#include "ProperOrthochronousLorentzGroup.hpp"
#include "ProperOrthochronousLorentzAlgebra.hpp"
#include "VectorField.hpp"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>

std::vector<Eigen::MatrixXd> generate_curve(int n_points, float radius, float h=1.0, float c=0.2) {
    // Creates a curve in SE(3) with 100 points. The curve is a circle where each
    // point has an orientation frame attached to it. The 'curve' in orientation space
    // is a rotation around the z-axis.
    std::vector<Eigen::MatrixXd> curve;
    for (int i = 0; i < n_points; i++) {
        // Create a point on the circle
        float angle = 2 * M_PI * i / n_points;
        Eigen::MatrixXd point = Eigen::MatrixXd::Identity(4, 4);
        point(0, 3) = radius * cos(angle);
        point(1, 3) = radius * sin(angle);
        point(2, 3) = h + c*(pow(radius*cos(angle), 2) - pow(radius*sin(angle), 2));
        
        // Curve option 2:
        // point(0, 3) = radius * (sin(angle) + 2*sin(2*angle));
        // point(1, 3) = radius * (cos(angle) - 2*cos(2*angle));
        // point(2, 3) = h - radius * sin(3*angle);
        
        // Create a rotation matrix around the z-axis
        point(1, 2) = sin(angle);
        point(1, 1) = cos(angle);
        point(2, 2) = cos(angle);
        point(2, 1) = -sin(angle);
        // Combine the point and the rotation
        curve.push_back(point);
    }
    return curve;
}

void writeToCSV(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, float>>& iterationResults, const std::string& filename) {
    std::ofstream csvFile;
    csvFile.open(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Loop over each iteration's result
    for (const auto& data : iterationResults) {
        Eigen::MatrixXd matrix = std::get<0>(data);
        // std::cout << "LOG-- matrix: " << std::endl << matrix << std::endl;
        Eigen::VectorXd tangent = std::get<1>(data);
        Eigen::VectorXd normal = std::get<2>(data);
        float distance = std::get<3>(data);
        // Flatten the matrix and write it to the CSV
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                csvFile << matrix(i, j);
                if (!(i == matrix.rows() - 1 && j == matrix.cols() - 1)) {
                    csvFile << ";";  // Separate matrix entries with semicolons
                }
            }
        }

        csvFile << ";";  // Separate matrix from vector components

        // Write the tangent vector
        for (int i = 0; i < tangent.size(); ++i) {
            csvFile << tangent(i);
            if (i != tangent.size() - 1) {
                csvFile << ";";  // Separate vector entries with semicolons
            }
        }

        csvFile << ";";  // Separate tangent from normal vector

        // Write the normal vector
        for (int i = 0; i < normal.size(); ++i) {
            csvFile << normal(i);
            if (i != normal.size() - 1) {
                csvFile << ";";  // Separate vector entries with semicolons
            }
        }

        // write the minimum distance
        csvFile << ";" << distance;

        csvFile << "\n";  // Newline for next iteration result
    }

    csvFile.close();
    std::cout << "Data successfully written to " << filename << std::endl;
}

void write_curve2csv(const std::vector<Eigen::MatrixXd>& curve, const std::string& filename) {
    std::ofstream csvFile;
    csvFile.open(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Loop over each point in the curve
    for (const auto& point : curve) {
        // Flatten the matrix and write it to the CSV
        for (int i = 0; i < point.rows(); ++i) {
            for (int j = 0; j < point.cols(); ++j) {
                csvFile << point(i, j);
                if (!(i == point.rows() - 1 && j == point.cols() - 1)) {
                    csvFile << ";";  // Separate matrix entries with semicolons
                }
            }
        }

        csvFile << "\n";  // Newline for next point in the curve
    }

    csvFile.close();
    std::cout << "Data successfully written to " << filename << std::endl;
}

void test_se3(){
  SE3 X = SE3();
  X.print();
  Eigen::MatrixXd mat(4, 4);
  mat << 0.5, 0.5, 0, 1,
         -0.5, 0.5, 0, 2,
         0, 0, 1, 3,
         0, 0, 0, 1;
  Eigen::VectorXd xi(6);
  xi << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd H(4, 4);
  H << 0, -3, 2, 4,
       3, 0, -1, 8,
       -2, 1, 0, 12,
       0, 0, 0, 0;
  se3 A2 = se3(H);
  std::cout << "print A2" << std::endl;
  A2.print();
  se3 A = X.algebra().S(xi);
  std::cout << "print A" << std::endl;
  A.print();
  std::cout << "print xi" << std::endl;
  // Eigen::Matrix3d debug = Eigen::Matrix3d::Identity();
  Eigen::VectorXd xi2 = X.algebra().invS(A);
  std::cout << "invS(A) = " << xi2 << std::endl;
  std::cout << "here" << std::endl;
}

void test_VectorField(){
    std::vector<Eigen::MatrixXd> curve = generate_curve(5000, 1);
    std::cout << "Curve first point: " << std::endl << curve[0] << std::endl;
    std::cout << "curve size: " << curve.size() << std::endl;
    std::cout << "ds: " << 1.0 / curve.size() << std::endl;
    VectorField vf = VectorField<SE3>(curve, 0.001, 0.001);
    // create a std::vector to store each 4x4matrix of the updated state
    std::vector<Eigen::MatrixXd> updated_states;

    Eigen::MatrixXd H0 = Eigen::MatrixXd::Identity(4, 4);
    H0(0, 3) = -2;
    H0(1, 3) = -1;
    H0(2, 3) = 0;
    H0(0, 0) = cos(M_PI / 4);
    H0(0, 1) = -sin(M_PI / 4);
    H0(1, 0) = sin(M_PI / 4);
    H0(1, 1) = cos(M_PI / 4);
    std::cout << "H0: " << std::endl << H0 << std::endl;
    SE3 state0 = SE3(H0);
    
    // float dist = vf.EEdistance(state0, curve[0]);
    // std::cout << "Euclidean distance between state0 and curve[0]: " << dist << std::endl;
    // float min_dist = vf.ECdistance(state0);
    // std::cout << "Minimum distance between state0 and curve: " << min_dist << std::endl;
    // Eigen::VectorXd psi = vf(state0);
    // std::cout << "Vector field at state0: " << std::endl << psi << std::endl;

    // Simulate system
    float dt = 0.01;
    float T = 5.0;
    float gain_N = 1.0;
    float gain_T = 1.0;
    int n_steps = T / dt;
    SE3 state = state0;
    updated_states.push_back(state.matrix());
    std::cout << "Sanity check H0: " << std::endl << state.matrix() << std::endl;
    std::cout << "Simulating system for " << n_steps << " steps." << std::endl;
    // Computes mean time for each iteration + standard deviation
    std::vector<double> iterationTimes;
    for (int i = 0; i < n_steps; i++) {
        // auto start = std::chrono::high_resolution_clock::now();
        // if (i < 5){
        //     std::cout << "Iteration " << i << std::endl;
        //     std::cout << "State matrix: " << std::endl << state.matrix() << std::endl;
        // }
        std::cout << "Call eval" << std::endl;
        Eigen::VectorXd xi = vf.eval(state, true, 1.0, gain_N, 1.0, 1.0, gain_T);
        std::cout << "Call S" << std::endl;
        se3 liealg = state.algebra().S(xi);
        std::cout << "Build next_mat" << std::endl;
        Eigen::MatrixXd next_mat = (liealg * dt).exp() * state.matrix();
        std::cout << "Build SE3" << std::endl;
        Eigen::MatrixXd debug_mat = (liealg * dt).exp();
        // print determinants:
        // std::cout << "Determinant of state: " << state.matrix().determinant() << std::endl;
        Eigen::Matrix3d R_state = state.matrix().block<3, 3>(0, 0);
        Eigen::Matrix3d R_debug = debug_mat.block<3, 3>(0, 0);
        // Checks R*R^T
        // std::cout << "R*R^T: " << std::endl << (R_state * R_state.transpose().eval() - Eigen::Matrix3d::Identity()).norm() << std::endl;
        // std::cout << "R_debug*R_debug^T: " << std::endl << (R_debug * R_debug.transpose().eval() - Eigen::Matrix3d::Identity()).norm() << std::endl;

        // std::cout << "Debug matrix: " << std::endl << debug_mat << std::endl;
        state = SE3(next_mat);
        updated_states.push_back(state.matrix());
        // std::chrono::duration<double> duration = end - start;
        // iterationTimes.push_back(duration.count());
    }

    // Save the iteration data to a CSV file
    writeToCSV(vf.iterationResults, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/NEW_vf_data.csv");
    write_curve2csv(curve, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/NEW_curve_data.csv");
    write_curve2csv(updated_states, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/NEW_iteration_data.csv");
}

int main(){
  test_VectorField();

  return 0;
}