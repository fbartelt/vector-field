#include "SpecialEuclideanGroup.h"
#include "SpecialEuclideanAlgebra.h"
#include "VectorField.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
// #include <vector>
#include <fstream>
#include <string>

using namespace std;

void writeToCSV(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>>& iterationResults, const std::string& filename) {
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

void test_SpecialEuclideanGroup() {
    SpecialEuclideanAlgebra s1 = SpecialEuclideanAlgebra(3);
    s1.print();
    Eigen::VectorXd xi = Eigen::VectorXd::LinSpaced(6, 1, 6);
    cout << "xi = " << endl << xi << endl;
    Eigen::VectorXd hi = Eigen::VectorXd::LinSpaced(4, 11, 14);
    cout << "hi = " << endl << hi << endl;
    Eigen::MatrixXd SL_XI = s1.SL(xi, 3);
    cout << "SL(xi, 3) = " << endl << SL_XI << endl;
    Eigen::VectorXd inv_SL_XI = s1.invSL(SL_XI, 3);
    cout << "invSL(SL(xi, 3), 3) = " << endl << inv_SL_XI << endl;
    Eigen::MatrixXd SR_HI = s1.SR(hi, 3);
    cout << "SR(hi, 3, 3) = " << endl << SR_HI << endl;
    
    Eigen::MatrixXd hdot1 = SL_XI * hi;
    Eigen::MatrixXd hdot2 = SR_HI * xi;
    cout << "SL(xi, 3) * hi = " << endl << hdot1 << endl;
    cout << "SR(hi, 3, 3) * xi = " << endl << hdot2 << endl;


    SpecialEuclideanGroup rand_se3 = SpecialEuclideanGroup(3);
    rand_se3.print();
    std::cout << rand_se3.matrix().inverse() * rand_se3.matrix() << std::endl;
}

std::vector<Eigen::MatrixXd> generate_curve(int n_points, float radius) {
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

void test_VectorField(){
    std::vector<Eigen::MatrixXd> curve = generate_curve(500, 1);
    std::cout << "Curve first point: " << std::endl << curve[0] << std::endl;
    std::cout << "curve size: " << curve.size() << std::endl;
    std::cout << "ds: " << 1.0 / curve.size() << std::endl;
    VectorField vf = VectorField(curve, 0.01);

    Eigen::MatrixXd H0 = Eigen::MatrixXd::Identity(4, 4);
    H0(0, 3) = -1;
    H0(1, 3) = 1;
    H0(2, 3) = 1;
    H0(0, 0) = cos(M_PI / 4);
    H0(0, 1) = -sin(M_PI / 4);
    H0(1, 0) = sin(M_PI / 4);
    H0(1, 1) = cos(M_PI / 4);
    SpecialEuclideanGroup state0 = SpecialEuclideanGroup(3, H0);
    
    float dist = vf.EEdistance(state0, curve[0]);
    std::cout << "Euclidean distance between state0 and curve[0]: " << dist << std::endl;
    float min_dist = vf.ECdistance(state0);
    std::cout << "Minimum distance between state0 and curve: " << min_dist << std::endl;
    Eigen::VectorXd psi = vf(state0);
    std::cout << "Vector field at state0: " << std::endl << psi << std::endl;

    // Simulate system
    float dt = 0.01;
    float T = 5;
    int n_steps = T / dt;
    SpecialEuclideanGroup state = state0;
    std::cout << "Simulating system for " << n_steps << " steps." << std::endl;
    for (int i = 0; i < n_steps; i++) {
        Eigen::VectorXd xi = vf.eval(state, true);
        Eigen::MatrixXd liealg = state.algebra_.SL(xi, state.n());
        Eigen::MatrixXd next_mat = (liealg * dt).exp() * state.matrix();
        state = SpecialEuclideanGroup(state.n(), next_mat);
    }

    // Save the iteration data to a CSV file
    writeToCSV(vf.iterationResults, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/iteration_data.csv");
    write_curve2csv(curve, "/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/curve_data.csv");
}

int main() {
    // test_SpecialEuclideanGroup();
    test_VectorField();
    return 0;
}
// Output:
// This is the se(3) Lie algebra.
// SL(xi, 3) = 
//  0  1  2
// -1  0  3
// -2 -3  0