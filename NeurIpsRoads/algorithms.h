#ifndef alogrithms_h
#define algorithms_h

#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <iostream>
/*#include <GPy/GPy.hpp>
#include <GPy/kern/RBF.hpp>
#include <Eigen/Dense>

using namespace Eigen;*/

using namespace std;

class Player_Hedge {
public:
    Player_Hedge(int K, double T, double min_payoff, double max_payoff);

    std::vector<double> mixed_strategy();
    int sample_action();
    void Update(std::vector<int> played_actions, int player_idx, std::vector<std::vector<double>> SiouxNetwork_data_original, std::vector<double> Capacities_t, std::vector<std::vector<double>> Strategy_vectors);
    // Funciones auxiliares 
    int get_K() const { return K_; }
    double get_T() const { return T_; }
    double get_min_payoff() const { return min_payoff_; }
    double get_max_payoff() const { return max_payoff_; }
    std::string to_string() const;


private:
    std::string type_;
    int K_;
    double min_payoff_;
    double max_payoff_;
    std::vector<double> weights_;
    double T_;
    double gamma_t_;
};



/*
class Player_cGPMW {
public:
    Player_cGPMW(int K, int T, double min_payoff, double max_payoff, VectorXd Capacities, MatrixXd my_strategy_vecs,
        GPy::RBF* kernel, double sigma_e, int version);

    std::string type;
    int K;
    double min_payoff;
    double max_payoff;
    VectorXd weights;
    int T;
    VectorXi idx_nonzeros;
    double gamma_t;
    GPy::RBF* kernel;
    double sigma_e;
    MatrixXd strategy_vecs;
    MatrixXd history;
    MatrixXd history_payoffs;
    MatrixXd history_played_actions;
    std::vector<VectorXd> history_occupancies;
    VectorXd contexts;
    VectorXi idx_balls;
    int version;
    VectorXd Capacities;

    VectorXd mixed_strategy();
    int sample_action();
    void Update_history(int played_action, double payoff, VectorXd occupancies, VectorXd capacities_t);
    void Compute_strategy(VectorXd capacities_t);
};*/

#endif
