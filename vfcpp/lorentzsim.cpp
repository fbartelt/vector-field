#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

#include "ProperOrthochronousLorentzAlgebra.hpp"
#include "ProperOrthochronousLorentzGroup.hpp"
#include "VectorField.hpp"

std::vector<Eigen::MatrixXd> genCurve(int n_points, double v = 0.9999) {
  std::vector<Eigen::MatrixXd> curve;
  for (int i = 0; i < n_points; i++) {
    // Create a point on the circle
    double s = 2 * M_PI * i / n_points;
    double beta_x = v * 0.5 * (cos(s) + 1);
    double gamma_x = 1 / sqrt(1 - pow(beta_x, 2));
    Eigen::MatrixXd boost_x(4, 4);
    // Default format is (x, y, z, t)
    boost_x << gamma_x, 0, 0, -gamma_x * beta_x, 
               0, 1, 0, 0, 
               0, 0, 1, 0,
               -gamma_x * beta_x, 0, 0, gamma_x;

    curve.push_back(boost_x);
  }
  return curve;
}

std::vector<Eigen::MatrixXd> genCurveDerivative(int n_points,
                                                double v = 0.9999) {
  std::vector<Eigen::MatrixXd> dcurve;
  for (int i = 0; i < n_points; i++) {
    // Create a point on the circle
    double s = 2.0 * M_PI * i / n_points;
    double beta_x = v * 0.5 * (cos(s) + 1);
    double gamma_x = 1.0 / sqrt(1 - pow(beta_x, 2));
    double dbeta_x = -v * 0.5 * sin(s);
    double dgamma_x = (beta_x / pow(1 - pow(beta_x, 2), 1.5)) * dbeta_x;
    Eigen::MatrixXd dboost_x(4, 4);
    // Default format is (x, y, z, t)
    dboost_x << dgamma_x, 0, 0, -dgamma_x * beta_x - gamma_x * dbeta_x, 0, 0, 0,
        0, 0, 0, 0, 0, -dgamma_x * beta_x - gamma_x * dbeta_x, 0, 0, dgamma_x;

    dcurve.push_back(dboost_x);
  }
  return dcurve;
}

void writeToCSV(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double>>& iterationResults, const std::string& filename) {
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
        double distance = std::get<3>(data);
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

bool checkSO31(const Eigen::MatrixXd& mat){
    Eigen::MatrixXd Ipq = Eigen::MatrixXd::Identity(4, 4);
    Ipq(3, 3) = -1;
    Eigen::MatrixXd matT = mat.transpose();
    if (((matT * Ipq * mat) - Ipq).norm() < 1e-6 && mat.determinant() > 0){
        if (mat(3, 3) > 0){
            return true;
        }
    }
    std::cout << "Det: " << mat.determinant() << std::endl;
    std::cout << "mat(3, 3): " << mat(3, 3) << std::endl;
    std::cout << "||.||: " << ((matT * Ipq * mat) - Ipq).norm() << std::endl;
    return false;
}

void test_VectorField(){
    int npoints = 15000;
    double vel=0.9999;
    double epsilon = 0.001;
    double ds = 0.001;
    std::vector<Eigen::MatrixXd> curve = genCurve(npoints, vel);
    std::cout << "Curve first point: " << std::endl << curve[0] << std::endl;
    std::cout << "curve size: " << curve.size() << std::endl;
    std::vector<Eigen::MatrixXd> dcurve = genCurveDerivative(npoints, vel);
    VectorField vf = VectorField<SO31>(curve, dcurve, epsilon, ds);
    // VectorField vf = VectorField<SO31>(curve, epsilon, ds);
    // create a std::vector to store each 4x4matrix of the updated state
    std::vector<Eigen::MatrixXd> updated_states;
    
    so31 aux = so31();
    Eigen::VectorXd xi0(6);
    xi0 << 0.0, 0, 0, 0.7, 0, 0;
    so31 liealg0 = aux.S(xi0);
    Eigen::MatrixXd H0 = liealg0.exp();
    // Eigen::MatrixXd H0 = Eigen::MatrixXd::Identity(4, 4);
    std::cout << "H0: " << std::endl << H0 << std::endl;
    SO31 state0 = SO31(H0);
    
    // double dist = vf.EEdistance(state0, curve[0]);
    // std::cout << "Euclidean distance between state0 and curve[0]: " << dist << std::endl;
    // double min_dist = vf.ECdistance(state0);
    // std::cout << "Minimum distance between state0 and curve: " << min_dist << std::endl;
    // Eigen::VectorXd psi = vf(state0);
    // std::cout << "Vector field at state0: " << std::endl << psi << std::endl;

    // Simulate system
    double dt = 0.01;
    double T = 20.0;
    double gain_N = 10000.0; // 1000.0
    double gain_T = 10000.0; // 100.0
    double gain_T1 = 0.5;
    int n_steps = T / dt;
    SO31 state = state0;
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
        // std::cout << "Call eval" << std::endl;
        Eigen::VectorXd xi = vf.eval(state, true, 1.0, gain_N, gain_T1, 1.0, gain_T);
        // std::cout << "Call S" << std::endl;
        so31 liealg = state.algebra().S(xi);
        // std::cout << "Build next_mat" << std::endl;
        Eigen::MatrixXd next_mat = (liealg * dt).exp() * state.matrix();
        // std::cout << "Build SE3" << std::endl;
        Eigen::MatrixXd debug_mat = (liealg * dt).exp();

        if(!(checkSO31(debug_mat))){
            std::cout << "Error: debug_mat is not in SO31" << std::endl;
            std::cout << debug_mat << std::endl;
        }
        if(!checkSO31(next_mat)){
            std::cout << "Error: next_mat is not in SO31" << std::endl;
            std::cout << next_mat << std::endl;
        }
        // print determinants:
        // std::cout << "Determinant of state: " << state.matrix().determinant() << std::endl;
        // Eigen::Matrix3d R_state = state.matrix().block<3, 3>(0, 0);
        // Eigen::Matrix3d R_debug = debug_mat.block<3, 3>(0, 0);
        // Checks R*R^T
        // std::cout << "R*R^T: " << std::endl << (R_state * R_state.transpose().eval() - Eigen::Matrix3d::Identity()).norm() << std::endl;
        // std::cout << "R_debug*R_debug^T: " << std::endl << (R_debug * R_debug.transpose().eval() - Eigen::Matrix3d::Identity()).norm() << std::endl;

        // std::cout << "Debug matrix: " << std::endl << debug_mat << std::endl;
        state = SO31(next_mat);
        updated_states.push_back(state.matrix());
        // std::chrono::duration<double> duration = end - start;
        // iterationTimes.push_back(duration.count());
    }

    // Save the iteration data to a CSV file
    writeToCSV(vf.iterationResults, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/LORENTZ_vf_data.csv");
    write_curve2csv(curve, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/LORENTZ_curve_data.csv");
    write_curve2csv(updated_states, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/LORENTZ_iteration_data.csv");

    // Write to csv a vector<double>:
    std::ofstream csvFile;
    csvFile.open("/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/LORENTZ_closeidx.csv");
    for (const auto& data : vf.closest_indices){
        csvFile << data << "\n";
    }
    csvFile.close();
}

int main(){
  test_VectorField();

  return 0;
}