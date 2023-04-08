#ifndef alogrithms_h
#define algorithms_h

#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <iostream>
#include <random>
#include "network.h"

/*
#include <GPy/GPy.hpp>
#include <GPy/kern/RBF.hpp>
#include <Eigen/Dense>

using namespace Eigen;*/


/*
 Player Hedge
*/

/*
IDEA: herencia player padre del resto de jugadores para tratar todos los jugadores en auxiliar.
plus: se podría crear un .h y un .cpp para player y otro para cada tipo.
*/

enum class PlayerType {
    cGPMW,
    Hedge,
    cGPMWpar
};

class Player {
protected:
    PlayerType type_;
    int K_;
    double min_payoff_;
    double max_payoff_;
    std::vector<double> weights_;
    double T_;
    double gamma_t_;

public:

    virtual int sample_action();
    virtual void Update(std::vector<int> played_actions, int player_idx, const NetworkData& network, std::vector<double> Capacities_t, std::vector<std::vector<std::vector<int>>> Strategy_vectors);

    int getK();
    PlayerType getType();

};


class Player_Hedge : public Player {
public:
    Player_Hedge(int K, double T, double min_payoff, double max_payoff) {
        this->K_ = K;
        this->T_ = T;
        this->min_payoff_ = min_payoff;
        this->max_payoff_ = max_payoff;
        this->gamma_t_ = (sqrt(8 * log(K) / T));// tasa aprendizaje
        //this->type_ = Playertype::Hedge;
        this->weights_ = std::vector<double>(K, 1); // para cada brazo el valor inicial en la distribución es 1
    }

    std::vector<double> mixed_strategy();
    int sample_action() override;
    void Update(std::vector<int> played_actions, int player_idx, const NetworkData& network, std::vector<double> Capacities_t, std::vector<std::vector<std::vector<int>>> Strategy_vectors);
    // Funciones auxiliares 
    int get_K() const { return K_; }
    double get_T() const { return T_; }
    double get_min_payoff() const { return min_payoff_; }
    double get_max_payoff() const { return max_payoff_; }
};

/*
Player GPMW
*//*

class Player_GPMW {
public:
    Player_GPMW(int K, int T, double min_payoff, double max_payoff, std::vector<std::vector<double>> my_strategy_vecs, double kernel, double sigma_e);

    std::vector<double> mixed_strategy();
    int sample_action();
    void Update(int played_action, std::vector<std::vector<double>> total_occupancies, std::vector<double> payoff, std::vector<double> Capacities_t);


private:
    std::string type;
    int K;
    double min_payoff;
    double max_payoff;
    std::vector<double> weights;
    int T;
    std::vector<int> idx_nonzeros;

    std::vector<double> cum_losses;
    std::vector<double> mean_rewards_est;
    std::vector<double> std_rewards_est;
    std::vector<double> ucb_rewards_est;
    double gamma_t;
    double kernel;
    double sigma_e;
    std::vector<std::vector<double>> strategy_vecs;

    std::vector<double> history_payoffs;
    std::vector<std::vector<double>> history;
    double demand;
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
