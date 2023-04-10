#ifndef GAUSSIAN_PROCESS_REGRESSION_H
#define GAUSSIAN_PROCESS_REGRESSION_H

#include <Eigen/Dense>

class GaussianProcessRegression {
public:
    GaussianProcessRegression(const Eigen::MatrixXd& kernel, double sigma_squared);

    void train(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> predict(const Eigen::MatrixXd& X_test);

private:
    Eigen::MatrixXd kernel_;
    double sigma_squared_;
    Eigen::MatrixXd X_train_;
    Eigen::VectorXd y_train_;
    Eigen::MatrixXd K_inv_;
};

#endif // GAUSSIAN_PROCESS_REGRESSION_H
