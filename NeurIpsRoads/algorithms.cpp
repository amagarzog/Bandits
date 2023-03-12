#include "algorithms.h"


Player_Hedge::Player_Hedge(int K, double T, double min_payoff, double max_payoff) :
    K_(K),
    min_payoff_(min_payoff),
    max_payoff_(max_payoff),
    weights_(K, 1),
    T_(T),
    gamma_t_(sqrt(8 * log(K) / T)) {
    type_ = "Hedge";
}

vector<double> Player_Hedge::mixed_strategy() {
    vector<double> strategy(K_);
    // strategy representa la probabilidad de elegir cada accion
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        strategy[i] = weights_[i] / sum_weights;
    }
    return strategy;
}

int Player_Hedge::sample_action() {
    vector<double> strategy = mixed_strategy();
    double r = ((double)rand() / RAND_MAX);
    double sum_prob = 0.0;
    for (int i = 0; i < K_; i++) {
        sum_prob += strategy[i];
        if (r <= sum_prob) {
            return i; // se elige la acción
        }
    }
    return K_ - 1; // In case of numerical errors
}


std::string Player_Hedge::to_string() const {
    std::string str = "Type: " + type_ + "\n" +
        "K: " + std::to_string(K_) + "\n" +
        "T: " + std::to_string(T_) + "\n" +
        "Min Payoff: " + std::to_string(min_payoff_) + "\n" +
        "Max Payoff: " + std::to_string(max_payoff_) + "\n";
    return str;
}


void Player_Hedge::Update(std::vector<int> played_actions, int player_idx, std::vector<std::vector<double>> SiouxNetwork_data_original, std::vector<double> Capacities_t, std::vector<std::vector<double>> Strategy_vectors) {
    std::vector<double> losses_hindsight(K_);
    for (int a = 0; a < K_; a++) {
        std::vector<int> modified_outcome = played_actions;
        modified_outcome[player_idx] = a;
        //perdidas si hubiera elegido el brazo a
        losses_hindsight[a] = 2; //Compute_traveltimes(SiouxNetwork_data_original, Strategy_vectors, modified_outcome, player_idx, Capacities_t);
    }

    // Se establecen recompensas por cada accion y se limitan al maximo payoff y minimo payoff
    std::vector<double> payoffs(K_);
    for (int i = 0; i < K_; i++) {
        payoffs[i] = -losses_hindsight[i];
        payoffs[i] = std::max(payoffs[i], min_payoff_);
        payoffs[i] = std::min(payoffs[i], max_payoff_);
    }

    // Se limitan los recompensas para cada brazo para que esten entre el intervalo 0 y 1
    std::vector<double> payoffs_scaled(K_);
    for (int i = 0; i < K_; i++) {
        payoffs_scaled[i] = (payoffs[i] - min_payoff_) / (max_payoff_ - min_payoff_);
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

    // Se normalizan los pesos
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        weights_[i] = weights_[i] / sum_weights;
    }
}











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