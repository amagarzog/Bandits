#ifndef auxiliar_h
#define auxiliar_h

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include "algorithms.h"

class GameData {
public:
    GameData(int N);
    std::vector<int> Played_actions;
    std::vector<std::vector<double>> Mixed_strategies;
    std::vector<double> Incurred_losses;
    std::vector<std::vector<double>> Regrets;
    std::vector<std::vector<double>> Cum_losses;
};

//void Initialize_Players(int N, vector<string> OD_pairs, vector<vector<vector<double>>> Strategy_vectors, vector<double> min_traveltimes, vector<double> max_traveltimes, vector<int> idxs_controlled, double T, string Algo, int version, vector<double> Sigma, vector<vector<double>> Kernels, vector<double> sigmas, int numberofcontexts, vector<vector<int>> Capacities, vector<Player*>& Players);

void Initilize_Players_ini();
#endif
