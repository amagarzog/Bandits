#ifndef auxiliar_h
#define auxiliar_h

#include <vector>
#include <string>
#include "algorithms.h"
using namespace std;

class GameData {
public:
    GameData(int N);
    vector<int> Played_actions;
    vector<vector<double>> Mixed_strategies;
    vector<double> Incurred_losses;
    vector<vector<double>> Regrets;
    vector<vector<double>> Cum_losses;
};

void Initialize_Players(int N, vector<string> OD_pairs, vector<vector<vector<double>>> Strategy_vectors, vector<double> min_traveltimes, vector<double> max_traveltimes, vector<int> idxs_controlled, double T, string Algo, int version, vector<double> Sigma, vector<vector<double>> Kernels, vector<double> sigmas, int numberofcontexts, vector<vector<int>> Capacities, vector<Player*>& Players);

#endif
