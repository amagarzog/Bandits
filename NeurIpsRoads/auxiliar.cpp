#include "auxiliar.h"


/*void Initialize_Players(int N, vector<string> OD_pairs, vector<vector<vector<double>>> Strategy_vectors, vector<double> min_traveltimes, vector<double> max_traveltimes, vector<int> idxs_controlled, double T, string Algo, int version, vector<double> Sigma, vector<vector<double>> Kernels, vector<double> sigmas, int numberofcontexts, vector<vector<int>> Capacities, vector<Player*>& Players) {
    for (int i = 0; i < N; i++) {
        int K_i = Strategy_vectors[i].size();
        double min_payoff = -max_traveltimes[i];
        double max_payoff = -min_traveltimes[i];
        // idxs son los ids de los agentes que son controlados
        /*if (find(idxs_controlled.begin(), idxs_controlled.end(), i) != idxs_controlled.end() && K_i > 1) { // si el agente está controlado por el agente y tiene más de un brazo
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

}*/


void Initilize_Players_ini() {

    cout << "Se inicializan a los jugadores" << endl;
    int K = 5;
    double T = 10.0;
    double min_payoff = -1.0;
    double max_payoff = 1.0;

    Player_Hedge hedge_player(K, T, min_payoff, max_payoff);
    std::cout << hedge_player.to_string() << std::endl;
    cout << "Accion " << hedge_player.sample_action() << endl;


}

/*
#include <vector>
#include <cmath>
#include <iostream>
#include "Game_data.hpp"
#include "Player.hpp"
#include "SiouxNetwork_data_original.hpp"
#include "Strategy_vectors.hpp"

void Simulate_Game(int run, std::vector<Player*>& Players, int T, SiouxNetwork_data_original& SiouxNetwork_data_original, Strategy_vectors& Strategy_vectors, std::vector<double>& sigmas, std::vector<std::vector<int>>& Capacities, std::vector<int>* Contexts = nullptr) {
    int N = Players.size();
    Game_data Game_data(N);
    for (int i = 0; i < N; ++i) {
        Game_data.Cum_losses[i].resize(Players[i]->K);
    }

    std::vector<std::vector<double>> Total_occupancies;
    std::vector<std::vector<double>> addit_Congestions;
    std::vector<double> original_capacities(SiouxNetwork_data_original.Capacities);

    for (int t = 0; t < T; ++t) {
        std::vector<int> Capacities_t(Capacities[Contexts != nullptr ? (*Contexts)[t] : 0]);
        std::vector<int> played_actions_t(N);
        for (int i = 0; i < N; ++i) {
            if (Players[i]->type == "cHedge" || Players[i]->type == "cGPMWpar" || Players[i]->type == "RobustLinExp3") {
                played_actions_t[i] = Players[i]->sample_action(Contexts != nullptr ? (*Contexts)[t] : 0, Capacities_t);
            } else {
                if (Players[i]->type == "cGPMW" && t > 0) {
                    Players[i]->Compute_strategy(Capacities_t);
                }
                played_actions_t[i] = Players[i]->sample_action();
            }
        }
        Game_data.Played_actions.push_back(played_actions_t);

        std::vector<double> losses_t = Compute_traveltimes(SiouxNetwork_data_original, Strategy_vectors, Game_data.Played_actions[t], "all", Capacities_t);
        Game_data.Incurred_losses.push_back(losses_t);

        Total_occupancies.push_back(std::vector<double>(Strategy_vectors[0].size(), 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < Strategy_vectors[i].size(); ++j) {
                Total_occupancies[t][j] += Strategy_vectors[i][Game_data.Played_actions[t][i]][j];
            }
        }

        std::vector<double> congestions(Capacities_t.size(), 0.0);
        for (int i = 0; i < Capacities_t.size(); ++i) {
            congestions[i] = 0.15 * std::pow(Total_occupancies[t][i] / Capacities_t[i], 4);
        }
        addit_Congestions.push_back(congestions);

        for (int i = 0; i < N; ++i) {
            if (Players[i]->type == "EXP3P") {
                double noisy_loss = Game_data.Incurred_losses[t][i] + sigmas[i] * std::sqrt(2 * std::log(N) / (N * T));
                Players[i]->Update(Game_data.Played_actions[t][i], -noisy_loss);

            } else if (Players[i].type == "Hedge") {
                Players[i].Update(Game_data.Played_actions[t], i , SiouxNetwork_data_original, Capacities_t,  Strategy_vectors);
            }

            if (Players[i].type == "cHedge") {
                Players[i].Update(Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, original_capacities, Capacities_t,  Strategy_vectors);
            }

            if (Players[i].type == "RobustLinExp3") {
                double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                Players[i].Update(Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, Strategy_vectors, original_capacities, Capacities_t, noisy_loss);
            }

            if (Players[i].type == "GPMW") {
                double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                Players[i].Update(Game_data.Played_actions[t][i], Total_occupancies.back(), -noisy_loss, Capacities_t);
            }

            if (Players[i].type == "cGPMW") {
                double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                Players[i].Update_history(Game_data.Played_actions[t][i], -noisy_loss, Total_occupancies.back(), Capacities_t );
            }

            if (Players[i].type == "cGPMWpar") {
                double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                Players[i].Update_history(Game_data.Played_actions[t][i], -noisy_loss, Total_occupancies.back(), Capacities_t );
                Players[i].Update( Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, original_capacities, Capacities_t,  Strategy_vectors);
            }
        }
        double avg_cong = 0;
        for (int i = 0; i < addit_Congestions.size(); i++) {
            double sum = 0;
            for (int j = 0; j < addit_Congestions[i].size(); j++) {
                sum += addit_Congestions[i][j];
            }
            avg_cong += sum / addit_Congestions[i].size();
        }

        avg_cong /= addit_Congestions.size();

        cout << Players[2].type << " run: " << run+1 << ", time: " << t << ", Avg cong. " << fixed << setprecision(2) << avg_cong << endl;

        return make_tuple(Game_data, Total_occupancies , addit_Congestions);



*/




/*


#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <GPy/kern.h>
#include <GPy/models/gp_regression.h>

using namespace Eigen;

typedef SparseMatrix<double> SpMat;
typedef Triplet<double> T;

std::tuple<std::vector<GPy::Kern*>, std::vector<MatrixXd>> Optimize_Kernels(bool reoptimize, std::string Algo, std::vector<GPy::Kern*> Kernels, std::vector<int> idxs_controlled, MatrixXd Strategy_vectors, VectorXd sigmas, int poly_degree, MatrixXd Outcomes, MatrixXd Capacities, MatrixXd Payoffs){
    int N = Kernels.size();
    MatrixXd list_of_param_arrays = np.load("list_of_param_arrays_" + Algo + ".npy");
    for (int p : idxs_controlled){
        if (Kernels[p] == nullptr){
            VectorXd idx_nonzeros = (Strategy_vectors.rowwise().sum() != 0).cast<double>().array();
            int dim = idx_nonzeros.sum();
            if (reoptimize == false){
                MatrixXd loaded_params = list_of_param_arrays.row(p);
                GPy::Kern* kernel_1 = new GPy::Poly(dim, 1, idx_nonzeros.cast<int>(), loaded_params(0), loaded_params(1), loaded_params(2));
                GPy::Kern* kernel_2 = new GPy::Poly(dim, poly_degree, idx_nonzeros.cast<int>() + dim, loaded_params(3), loaded_params(4), loaded_params(5));
                Kernels[p] = kernel_1->operator*(*kernel_2);
            }
            if (reoptimize == true){
                GPy::Kern* kernel_1 = new GPy::Poly(dim, 1, idx_nonzeros.cast<int>(), 1, 1e-6);
                GPy::Kern* kernel_2 = new GPy::Poly(dim, poly_degree, idx_nonzeros.cast<int>() + dim);
                Kernels[p] = kernel_1->operator*(*kernel_2);

                if (Strategy_vectors.rows() > 1){
                    MatrixXd X(0, dim * 2);
                    MatrixXd y(0, 1);
                    MatrixXd y_true(0, 1);
                    for (int a = 0; a < 500; a++){
                        VectorXd x1 = Strategy_vectors.row(a).segment(idx_nonzeros.cast<int>().array(), idx_nonzeros.sum());
                        VectorXd occupancies = Strategy_vectors.colwise().sum().array();
                        if (Algo == "GPMW"){
                            x1.conservativeResize(x1.size() + 1);
                            x1(x1.size() - 1) = occupancies(p);
                        }
                        else{
                            x1 = x1.cwiseQuotient(Capacities.row(a)).array();
                        }
                        X.conservativeResize(X.rows() + 1, X.cols());
                        X.row(X.rows() - 1) << x1.transpose(), x1.transpose();
                        y.conservativeResize(y.rows() + 1, 1);
                        y(y.rows() - 1, 0) = Payoffs(a, p) + 1 * std::normal_distribution<double>(0, sigmas(p))(std::mt19937(std::random_device()()));
                        y_true.conservativeResize(y_true.rows() + 1, 1);
                        y_true(y.conservativeResize(y_true.rows() + 1, 1));
                        y_true(y_true.rows() - 1, 0) = Payoffs(a, p);
                    }
                    // Fit to data using Maximum Likelihood Estimation of the parameters
                    auto m = GPy::MakeRegressionModel(X.topRows(450), y.topRows(450), Kernels[p]);
                    m.GetGaussianNoise()->Fix(sigmas(p) * sigmas(p));
                    m.GetKernel().GetSubkernel(0).GetParam("bias").Fix();
                    m.GetKernel().GetSubkernel(0).GetParam("variance").Fix();
                    m.GetKernel().GetSubkernel(1).ConstrainBounded(1e-6, 1e6);
                    m.Optimize(max_f_eval = 100);

                    if (0) {
                        auto mu_var = m.Predict(X.bottomRows(50));
                        auto mu = mu_var.first;
                        auto var = mu_var.second;
                        auto sigma = (var.array().max(1e-6)).sqrt();

                        plt::plot(-y_true.bottomRows(50).array());
                        plt::plot(-mu.array());
                        plt::plot(-(mu + sigma).array());
                        auto ax = plt::gca();
                        ax->set_yscale("log");
                        //ax->set_ylim({-1000, 0});
                        plt::show();

                        plt::plot((y_true.bottomRows(50).array() - mu.array()).abs() / (y_true.bottomRows(50).array().abs()));
                        plt::show();
                    }
                }
            }
        }
    }

    if (reoptimize) { // override existing parameters
        std::vector<MatrixXd> list_of_param_arrays;
        for (int i = 0; i < Kernels.size(); ++i) {
            st_of_param_arrays.push_back(Kernels[i].GetParamArray());
        }
        np.save("list_of_param_arrays_" + Algo, list_of_param_arrays);
    }

    return Kernels, list_of_param_arrays;
}
*/