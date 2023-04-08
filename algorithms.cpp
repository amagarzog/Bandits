#include "algorithms.h"


std::vector<double> Player_Hedge::mixed_strategy() {
    std::vector<double> strategy(K_);
    // strategy representa la probabilidad de elegir cada accion
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        strategy[i] = weights_[i] / sum_weights; // de esa forma se consigue que cada estrategia/brazo tenga un aprobabilidad de ser elegida entre 0 y 1 de forma ponderada con el resto de brazos
    }
    return strategy;
}

int Player_Hedge::sample_action() {
    std::vector<double> strategy = mixed_strategy();
    double r = ((double)rand() / RAND_MAX); // de esta forma se consigue que r tenga un valor entre 0 y 1
    double sum_prob = 0.0;
    for (int i = 0; i < K_; i++) {
        sum_prob += strategy[i];
        if (r <= sum_prob) {
            return i; // se elige la acción
        }
    }
    return K_ - 1; // In case of numerical errors
}

void Player_Hedge::Update(std::vector<int> played_actions, int player_idx, const NetworkData &network, std::vector<double> Capacities_t, std::vector<std::vector<std::vector<int>>> Strategy_vectors) {
    std::vector<double> losses_hindsight(K_);
    for (int a = 0; a < K_; a++) {
        std::vector<int> modified_outcome = played_actions;
        modified_outcome[player_idx] = a; // modifica el brazo jugado a para ver cual seri ael travel time para los 5 brazos
        //perdidas si hubiera elegido el brazo a
        std::vector<double> traveltimetmp = Compute_traveltimes(network, Strategy_vectors, modified_outcome, player_idx, Capacities_t);
        losses_hindsight[a] = traveltimetmp[player_idx];
    }

    // Se establecen recompensas por cada accion y se limitan al maximo payoff y minimo payoff
    std::vector<double> payoffs(K_);
    for (int i = 0; i < K_; i++) {
        payoffs[i] = -losses_hindsight[i]; // las recompensas son los tiempos de viaje en negativo (cuanto menos tiempo de viaje, mejor recompensa)
        payoffs[i] = std::max(payoffs[i], min_payoff_);
        payoffs[i] = std::min(payoffs[i], max_payoff_); // delimitar entre min y maximo
    }

    // Se limitan los recompensas para cada brazo para que esten entre el intervalo 0 y 1
    std::vector<double> payoffs_scaled(K_);
    for (int i = 0; i < K_; i++) {
        payoffs_scaled[i] = (payoffs[i] - min_payoff_) / (max_payoff_ - min_payoff_); // - / - = +
    }

    // Se calculan las pérdidas sobre 1 - recompensas
    std::vector<double> losses(K_);
    for (int i = 0; i < K_; i++) {
        losses[i] = 1 - payoffs_scaled[i];
    }

    // Se actualizan los pesos en función de gamma y las pérdidas
    for (int i = 0; i < K_; i++) {
        weights_[i] = weights_[i] * exp(-gamma_t_ * losses[i]);
    }
    /*
    Si la pérdida es alta, entonces exp(-gamma_t_ * losses[i]) será un número pequeño y el nuevo peso w_i' será menor que el peso anterior w_i. Por otro lado, si la pérdida es baja o cero, 
    entonces exp(-gamma_t_ * losses[i]) será cercano a 1, lo que significa que el peso de la estrategia no cambia mucho en esa iteración. En general, el factor de actualización exponencial se 
    utiliza para ajustar la estrategia del jugador en función de su desempeño anterior, con el objetivo de mejorar su rendimiento en iteraciones futuras del juego.
    
    La tasa de aprendizaje hará que el peso se actualice con mayor o menor fuerza respecto a las pérdidas:
    Si la tasa de aprendizaje es alta las losses tendrán mas efecto en los weights ya que el exponencial sera un numero mayor.
    */

    // Se normalizan los pesos
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        weights_[i] = weights_[i] / sum_weights;
    }
}

/*
Player_GPMW::Player_GPMW(int K, int T, double min_payoff, double max_payoff, std::vector<std::vector<double>> my_strategy_vecs, double kernel, double sigma_e) {
    type = "GPMW";
    this->K = K;
    this->min_payoff = min_payoff;
    this->max_payoff = max_payoff;
    weights.resize(K, 1.0);
    this->T = T;

    for (int i = 0; i < my_strategy_vecs[0].size(); i++) {
        if (std::abs(std::accumulate(my_strategy_vecs.begin(), my_strategy_vecs.end(), 0.0, [i](double sum, std::vector<double> v) {return sum + v[i]; })) > 1e-5) {
            idx_nonzeros.push_back(i);
        }
    }

    cum_losses.resize(K, 0.0);
    mean_rewards_est.resize(K, 0.0);
    std_rewards_est.resize(K, 0.0);
    ucb_rewards_est.resize(K, 0.0);
    gamma_t = std::sqrt(8 * std::log(K) / T);
    this->kernel = kernel;
    this->sigma_e = sigma_e;
    strategy_vecs = my_strategy_vecs;

    history_payoffs = std::vector<double>();
    history = std::vector<std::vector<double>>();
    demand = *std::max_element(my_strategy_vecs[0].begin(), my_strategy_vecs[0].end());
}

std::vector<double> Player_GPMW::mixed_strategy() {
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::vector<double> mixed(K);
    for (int i = 0; i < K; i++) {
        mixed[i] = weights[i] / sum_weights;
    }
    return mixed;
}

int Player_GPMW::sample_action() {
    std::vector<double> mixed = mixed_strategy();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(mixed.begin(), mixed.end());
    return dist(gen);
}

/*

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "GPy/GPy.hpp"
#include "GPy/kern/GaussianKernel.h"


void Update(int played_action, double** total_occupancies, double* payoff, double* Capacities_t) {
    double beta_t = 0.5;
    int K = sizeof(self.strategy_vecs) / sizeof(self.strategy_vecs[0]);

    // Append new payoff to history_payoffs
    double** new_history_payoffs = new double*[self.history_payoffs_row + 1];
    for(int i=0; i<self.history_payoffs_row; i++) {
        new_history_payoffs[i] = new double[self.K];
        for(int j=0; j<self.K; j++) {
            new_history_payoffs[i][j] = self.history_payoffs[i][j];
        }
    }
    new_history_payoffs[self.history_payoffs_row] = payoff;
    self.history_payoffs = new_history_payoffs;
    self.history_payoffs_row += 1;

    // Append new row to history
    double** new_history = new double*[self.history_row + 1];
    for(int i=0; i<self.history_row; i++) {
        new_history[i] = new double[2*self.idx_nonzeros];
        for(int j=0; j<2*self.idx_nonzeros; j++) {
            new_history[i][j] = self.history[i][j];
        }
    }
    double* temp = new double[2*self.idx_nonzeros];
    for(int i=0; i<self.idx_nonzeros; i++) {
        temp[i] = self.strategy_vecs[played_action][self.idx_nonzeros][i];
        temp[self.idx_nonzeros+i] = total_occupancies[self.idx_nonzeros][i];
    }
    new_history[self.history_row] = temp;
    self.history = new_history;
    self.history_row += 1;

    // Create GP Regression model and fix Gaussian noise
    auto X = Eigen::Map<Eigen::MatrixXd>(self.history[0], self.history_row, 2*self.idx_nonzeros);
    auto y = Eigen::Map<Eigen::MatrixXd>(self.history_payoffs[0], self.history_payoffs_row, self.K);
    auto kernel = GPy::RBF(X.cols());
    auto m = GPy::GPRegression(X, y, kernel);
    m.Gaussian_noise.fix(std::pow(self.sigma_e, 2));

    // Compute UCB and mean/std estimates for each action
    double** other_occupancies = new double*[self.idx_nonzeros];
    for(int i=0; i<self.idx_nonzeros; i++) {
        other_occupancies[i] = new double[self.K];
        for(int j=0; j<self.K; j++) {
            other_occupancies[i][j] = total_occupancies[self.idx_nonzeros][j] - self.strategy_vecs[played_action][self.idx_nonzeros][j];
        }
    }
    double* mu = new double[self.K];
    double* var = new double[self.K];
    double* sigma = new double[self.K];
    for(int a1=0; a1<K; a1++) {
        double* x1 = new double[self.idx_nonzeros];
        for(int i=0; i<self.idx_nonzeros; i++) {
            x1[i] = self.strategy_vecs[a1][self.idx_nonzeros][i];
        }
        double* x2 = new double[self.K];
        for(int i=0; i<self.K; i++) {
            x2[i] = other_occupancies[i][j] + x1[j];
        }
        auto predict_input = Eigen::Map<Eigen::VectorXd>(std::vector<double>(x1, x1 + self.idx_nonzeros).data(), self.idx_nonzeros + self.K);
        predict_input.segment(self.idx_nonzeros, self.K) = Eigen::MapEigen::VectorXd(x2, self.K);
        Eigen::VectorXd mu, var;
        std::tie(mu, var) = m.predict(predict_input);

        double sigma = std::sqrt(std::max(var(0), 1e-6));
        self.ucb_rewards_est[a1] = mu(0) + beta_t * sigma;
        self.mean_rewards_est[a1] = mu(0);
        self.std_rewards_est[a1] = sigma;

        delete[] x1;
        delete[] x2;
    }

    double* payoffs = new double[K];
    for(int a1=0; a1<K; a1++) {
        payoffs[a1] = self.ucb_rewards_est[a1];
        payoffs[a1] = std::max(payoffs[a1], self.min_payoff);
        payoffs[a1] = std::min(payoffs[a1], self.max_payoff);
    }
    double* payoffs_scaled = new double[K];
    for(int a1=0; a1<K; a1++) {
        payoffs_scaled[a1] = (payoffs[a1] - self.min_payoff) / (self.max_payoff - self.min_payoff);
    }
    double* losses = new double[K];
    for(int a1=0; a1<K; a1++) {
        losses[a1] = 1.0 - payoffs_scaled[a1];
        self.cum_losses[a1] += losses[a1];
    }
    double* exp_losses = new double[K];
    for(int a1=0; a1<K; a1++) {
        exp_losses[a1] = std::exp(-gamma_t * self.cum_losses[a1]);
    }

    double sum_exp_losses = 0.0;
    for(int a1=0; a1<K; a1++) {
        sum_exp_losses += exp_losses[a1];
    }

    for(int a1=0; a1<K; a1++) {
        self.weights[a1] = exp_losses[a1] / sum_exp_losses;
    }

    delete[] payoffs;
    delete[] payoffs_scaled;
    delete[] losses;
    delete[] exp_losses;
}


*/











/*
Player_cGPMW::Player_cGPMW(int K, int T, double min_payoff, double max_payoff, VectorXd Capacities, MatrixXd my_strategy_vecs,
    GPy::RBF* kernel, double sigma_e, int version) {
    this->type = "cGPMW";
    this->K = K;
    this->min_payoff = min_payoff;
    this->max_payoff = max_payoff;
    this->weights = VectorXd::Ones(K);
    this->T = T;
    this->idx_nonzeros = VectorXi::Zero(my_strategy_vecs.cols());
    for (int i = 0; i < my_strategy_vecs.cols(); i++) {
        if (my_strategy_vecs.col(i).sum() != 0) {
            this->idx_nonzeros(i) = 1;
        }
    }

    this->gamma_t = sqrt(8 * log(K) / T);
    this->kernel = kernel;
    this->sigma_e = sigma_e;
    this->strategy_vecs = my_strategy_vecs;

    this->history_payoffs.resize(0, 1);
    this->history_played_actions.resize(0, 1);
    this->contexts = VectorXd::Zero(my_strategy_vecs.cols());
    this->idx_balls = VectorXi();
    this->version = version;
    this->Capacities = Capacities;
}

VectorXd Player_cGPMW::mixed_strategy() {
    return this->weights / this->weights.sum();
}

int Player_cGPMW::sample_action() {
    VectorXd mixed_strat = mixed_strategy();
    std::discrete_distribution<> dist(mixed_strat.data(), mixed_strat.data() + mixed_strat.size());
    return dist(GPy::Util::get_generator());
}
void Player_cGPMW::Update_history(int played_action, double payoff, Eigen::VectorXd occupancies, Eigen::VectorXd capacities) {
    history_played_actions.conservativeResize(history_played_actions.rows() + 1, history_played_actions.cols());
    history_payoffs.conservativeResize(history_payoffs.rows() + 1, history_payoffs.cols());
    history_played_actions(history_played_actions.rows() - 1) = played_action;
    history_payoffs(history_payoffs.rows() - 1) = payoff;

    Eigen::VectorXd occupancies_nonzeros(idx_nonzeros.size());
    Eigen::VectorXd capacities_nonzeros(idx_nonzeros.size());
    for (int i = 0; i < idx_nonzeros.size(); ++i) {
        occupancies_nonzeros(i) = occupancies(idx_nonzeros(i));
        capacities_nonzeros(i) = capacities(idx_nonzeros(i));
    }
    history_occupancies.push_back(occupancies_nonzeros);
    history.conservativeResize(history.rows() + 1, history.cols());
    Eigen::VectorXd row(history.cols());
    for (int i = 0; i < idx_nonzeros.size(); ++i) {
        row(i) = strategy_vecs(played_action)(idx_nonzeros(i));
        row(i + idx_nonzeros.size()) = occupancies_nonzeros(i) / capacities_nonzeros(i);
    }
    history(history.rows() - 1, Eigen::all) = row;
}
*/

int Player::getK()
{
    return this->K_;
}
