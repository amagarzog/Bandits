#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <gpy/gp.h>
#include <gpy/mean_functions.h>
#include <gpy/kernels.h>

class Player_cGPMW {
private:
    std::string type = "cGPMW";
    int K;
    double min_payoff, max_payoff;
    std::vector<double> weights;
    int T;
    std::vector<int> idx_nonzeros;
    double gamma_t;
    GPy::Kernel* kernel;
    double sigma_e;
    std::vector<std::vector<double>> strategy_vecs;
    std::vector<double> history_payoffs;
    std::vector<int> history_played_actions;
    std::vector<std::vector<double>> history_occupancies;
    std::vector<std::vector<double>> history;
    double demand;
    std::vector<double> contexts;
    std::vector<int> idx_balls;
    int version;
    std::vector<double> Capacities;

public:
    Player_cGPMW(int K, int T, double min_payoff, double max_payoff, std::vector<double> Capacities, std::vector<std::vector<double>> my_strategy_vecs, GPy::Kernel* kernel, double sigma_e, int version) {
        this->K = K;
        this->min_payoff = min_payoff;
        this->max_payoff = max_payoff;
        this->weights.resize(K, 1.0);
        this->T = T;
        this->idx_nonzeros = {};
        for (int i = 0; i < my_strategy_vecs[0].size(); i++) {
            if (std::accumulate(my_strategy_vecs.begin(), my_strategy_vecs.end(), 0.0, [&](double s, std::vector<double> v) { return s + v[i]; }) != 0.0) {
                this->idx_nonzeros.push_back(i);
            }
        }
        this->gamma_t = std::sqrt(8 * std::log(K) / T);
        this->kernel = kernel;
        this->sigma_e = sigma_e;
        this->strategy_vecs = my_strategy_vecs;
        this->history_payoffs = {};
        this->history_played_actions = {};
        this->history_occupancies = {};
        this->history = {};
        this->demand = *std::max_element(my_strategy_vecs[0].begin(), my_strategy_vecs[0].end());
        this->contexts = {};
        this->idx_balls = {};
        this->version = version;
        this->Capacities = Capacities;
    }

    std::vector<double> mixed_strategy() {
        double sum_weights = std::accumulate(this->weights.begin(), this->weights.end(), 0.0);
        std::vector<double> mixed(K);
        std::transform(this->weights.begin(), this->weights.end(), mixed.begin(), [&](double w) { return w / sum_weights; });
        return mixed;
    }

    int sample_action() {
        std::vector<double> mixed = mixed_strategy();
        std::discrete_distribution<int> dist(mixed.begin(), mixed.end());
        std::random_device rd;
        std::mt19937 gen(rd());
        return dist(gen);
    }

    void Update_history(int played_action, double payoff, std::vector<double> occupancies, std::vector<double> capacities) {
        this->history_played_actions.push(played_action);
        this->history_payoffs.push(payoff);
        this->history_occupancies.push(occupancies);
        this->history_capacities.push(capacities);
    }

    void compute_strategy() {
        double total_weight = 0.0;
        int num_actions = this->num_actions();
        std::vector<double> cumulative_regrets(num_actions, 0.0);

        // Compute positive regrets
        for (int i = 0; i < this->history_played_actions.size(); i++) {
            int played_action = this->history_played_actions[i];
            double weight = 1.0;
            for (int j = 0; j < i; j++) {
                weight *= this->history_occupancies[j][this->history_played_actions[j]];
            }
            cumulative_regrets[played_action] += weight * this->history_payoffs[i];
            total_weight += weight;
        }

        // Normalize cumulative regrets to obtain average strategy
        for (int i = 0; i < num_actions; i++) {
            if (total_weight > 0) {
                this->strategy[i] = fmax(cumulative_regrets[i] / total_weight, 0.0);
            }
            else {
                this->strategy[i] = 1.0 / num_actions;
            }
        }

        // Update capacity vector
        for (int i = 0; i < num_actions; i++) {
            this->capacity[i] = fmax(this->capacity[i] + this->capacity_scaling_factor * (this->strategy[i] - this->capacity[i]), 0.0);
        }
    }
