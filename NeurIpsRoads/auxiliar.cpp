#include "auxiliar.h"


void Initialize_Players(int N, vector<string> OD_pairs, vector<vector<vector<double>>> Strategy_vectors, vector<double> min_traveltimes, vector<double> max_traveltimes, vector<int> idxs_controlled, double T, string Algo, int version, vector<double> Sigma, vector<vector<double>> Kernels, vector<double> sigmas, int numberofcontexts, vector<vector<int>> Capacities, vector<Player*>& Players) {
    for (int i = 0; i < N; i++) {
        int K_i = Strategy_vectors[i].size();
        double min_payoff = -max_traveltimes[i];
        double max_payoff = -min_traveltimes[i];
        // idxs son los ids de los agentes que son controlados
        if (find(idxs_controlled.begin(), idxs_controlled.end(), i) != idxs_controlled.end() && K_i > 1) { // si el agente está controlado por el agente y tiene más de un brazo
            if (Algo == "Hedge") {
                Players[i] = new Player_Hedge(K_i, T, min_payoff, max_payoff);
            }
            else if (Algo == "cHedge") {
                Players[i] = new Player_cHedge(K_i, T, min_payoff, max_payoff, Capacities[i][0], numberofcontexts, Strategy_vectors[i], version);
            }
            else if (Algo == "cGPMWpar") {
                Players[i] = new Player_cGPMWpar(K_i, T, min_payoff, max_payoff, Capacities[i][0], numberofcontexts, Strategy_vectors[i], version, sigmas[i], Kernels[i]);
            }
            else if (Algo == "EXP3P") {
                Players[i] = new Player_EXP3P(K_i, T, min_payoff, max_payoff);
            }
            else if (Algo == "RobustLinExp3") {
                Players[i] = new Player_RobustLinExp3(K_i, T, min_payoff, max_payoff, Capacities[i][0], numberofcontexts, Strategy_vectors[i], Sigma, version);
            }
            else if (Algo == "GPMW") {
                Players[i] = new Player_GPMW(K_i, T, min_payoff, max_payoff, Strategy_vectors[i][0], Kernels[i][0], sigmas[i]);
            }
            else if (Algo == "cGPMW") {
                Players[i] = new Player_cGPMW(K_i, T, min_payoff, max_payoff, Capacities[i][0], Strategy_vectors[i][0], Kernels[i][0], sigmas[i], version);
            }
        }
        else {
            K_i = 1;
            Players[i] = new Player_Hedge(K_i, T, min_payoff, max_payoff);
        }
        Players[i]->OD_pair = OD_pairs[i];
    }
}
