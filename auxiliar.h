#ifndef auxiliar_h
#define auxiliar_h

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <string>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "algorithms.h"

class GameData {
public:
    GameData(int N);
    std::vector<std::vector<int>> Played_actions;
    std::vector<std::vector<double>> Mixed_strategies;
    std::vector<double> Incurred_losses;
    std::vector<std::vector<double>> Regrets;
    std::vector<std::vector<double>> Cum_losses;
};


// clases a implementar-> luego borrar
class SiouxNetwork_data_original {};
class Strategy_vectors {};

void Initialize_Players(int N, std::vector<std::string> OD_pairs, std::vector<std::vector<std::vector<double>>> Strategy_vectors, std::vector<double> min_traveltimes, std::vector<double> max_traveltimes, std::vector<int> idxs_controlled, double T, std::string Algo, int version, std::vector<double> Sigma, std::vector<std::vector<double>> Kernels, std::vector<double> sigmas, int numberofcontexts, std::vector<std::vector<int>> Capacities, std::vector<Player*>& Players);

void Initilize_Players_ini();
GameData Simulate_Game(int run, std::vector<Player> Players, int T, SiouxNetwork_data_original& SiouxNetwork_data_original, Strategy_vectors& Strategy_vectors, std::vector<double>& sigmas, std::vector<std::vector<int>>& Capacities, std::vector<std::vector<double>>& Total_occupancies, std::vector<std::vector<double>>& addit_Congestions, std::vector<int>* Contexts = nullptr);
std::vector<Eigen::MatrixXd> Optimize_Kernels(bool reoptimize, std::string Algo, const std::vector<int> &idxs_controlled, const std::vector<std::vector<std::vector<int>>>& Strategy_vectors, const std::vector<double> &sigmas, int poly_degree, const std::vector<std::vector<double>> &Outcomes, const std::vector<std::vector<double>> & Capacities, const std::vector<std::vector<double>> &Payoffs, std::vector<int> &listParmArrays);


#endif
