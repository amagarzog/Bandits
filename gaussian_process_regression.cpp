#include "gaussian_process_regression.h"
#include <Eigen/Cholesky>

GaussianProcessRegression::GaussianProcessRegression(const Eigen::MatrixXd& kernel, double sigma_squared)
    : kernel_(kernel), sigma_squared_(sigma_squared) {
}

void GaussianProcessRegression::train(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train) {
    assert(X_train.rows() <= kernel_.rows() / 2);
    X_train_ = X_train;
    y_train_ = y_train;

    int n = X_train.rows();
    Eigen::MatrixXd K = kernel_.block(0, 0, n, n) + sigma_squared_ * Eigen::MatrixXd::Identity(n, n);
    K_inv_ = K.llt().solve(Eigen::MatrixXd::Identity(n, n)); // Cholesky decomposition for inversion
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> GaussianProcessRegression::predict(const Eigen::MatrixXd& X_test) {
    assert(X_test.rows() == 1);
    assert(X_test.cols()/2 <= kernel_.cols());

    int n_train = X_train_.rows();
    int n_test = X_test.rows();

    Eigen::MatrixXd K_s(n_test, n_train);
    Eigen::MatrixXd K_ss(n_test, n_test);

    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_train; ++j) {
            K_s(i, j) = kernel_(i, j);
        }
    }

    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_test; ++j) {
            K_ss(i, j) = kernel_(i + n_train, j + n_train);
        }
    }

    Eigen::VectorXd mean = K_s * K_inv_ * y_train_;
    Eigen::MatrixXd covariance = K_ss - K_s * K_inv_ * K_s.transpose();

    // Use the diagonal of the covariance matrix for variance
    Eigen::VectorXd variance = covariance.diagonal();

    return std::make_pair(mean, variance);
}
