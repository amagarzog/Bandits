#include "auxiliar.h"

GameData:: GameData(int N, int T){
    this->Played_actions = std::vector<std::vector<int>>(T);
    this->Mixed_strategies = std::vector<double>(N); 
    this->Incurred_losses = std::vector<std::vector<double>>(T);
    this->Regrets = std::vector<double>(N);
    this->Cum_losses = std::vector<std::vector<double>>(N);
}

void GameData::Simulate_Game(int run, std::vector<Player*>& Players, int T, const NetworkData& network, std::vector<std::vector<std::vector<int>>>& Strategy_vectors, std::vector<double>& sigmas, std::vector<std::vector<double>>& Capacities, std::vector<std::vector<double>>& Total_occupancies, std::vector<std::vector<double>>& addit_Congestions, const std::vector<int>& Contexts)
{
    int N = Players.size();

    for (int i = 0; i < N; ++i) {
        // cum loses sera N X K, hay cum loses para cada brazo
        int brazosJugadorI = Players[i]->getK();
        this->Cum_losses[i].resize(brazosJugadorI);
    }


    std::vector<double> original_capacities = getCapacities(network);

    for (int t = 0; t < T; ++t) {
        std::vector<double> Capacities_t = Capacities[Contexts[t]]; // se coge la capacidad en la ronda t en funcion del contexto en esa ronda
        std::vector<int> played_actions_t(N);

        // 1 - Cada jugador juega una acción
        for (int i = 0; i < N; ++i) {
            if (Players[i]->getType() == PlayerType::cGPMW && t > 0) {
                //Players[i]->Compute_strategy(Capacities_t); // compute strategy es del cGPMW
            }
            played_actions_t[i] = Players[i]->sample_action();  // Los jugadores no controlados van a usar siempre su único brazo que es el 0
        }
        
        
        this->Played_actions[t] = played_actions_t; // Guarda las acciones de todos los jugadores de la ronda t

        // 2 - Asignar Recompensas/losses

        int identificador = -1; // Se asigna el identificador -1 para que compute travel times calcule los tiempos para todos los jugadores
        std::vector<double> losses_t = Compute_traveltimes(network, Strategy_vectors, this->Played_actions[t], identificador, Capacities_t);
        this->Incurred_losses[t] = losses_t ;

        int E = Strategy_vectors[0][0].size(); // numero de carreteras
        Total_occupancies.push_back(std::vector<double>(E, 0.0)); 
        
        /* Cada occupancies (cada ronda) guarda lo que se ocupa cada carretera sumando 
        * lo que usan los jugadores estas carreteras mediante las estrategias o brazos elegidos
        */
        std::vector<double> congestions(E, 0.0);

        for (int i = 0; i < N; ++i) {
            int aux = Strategy_vectors[i][this->Played_actions[t][i]].size();
            for (int j = 0; j < aux; ++j) { // 76 carreteras: sumar lo que ocupa en total la estrategia de cada jugador en las 76 carreteras
                Total_occupancies[t][j] += Strategy_vectors[i][this->Played_actions[t][i]][j]; // en cada total occupancies se guarda todo lo que ocupa un jugador en todas sus carreteras del brazo

            }

            for (int e = 0; e < Capacities_t.size(); ++e) {
                congestions[e] = 0.15 * std::pow(Total_occupancies[t][e] / Capacities_t[e], 4);
            }

        }
        addit_Congestions[t] = congestions;
            
        // 3 - Actualizar estrategias
        for (int i = 0; i < N; ++i) {

            if (Players[i]->getType() == PlayerType::Hedge) {
                if(Players[i]->getK() > 1)
                    Players[i]->Update(this->Played_actions[t], i, network, Capacities_t, Strategy_vectors);
            }
            if (Players[i]->getType() == PlayerType::GPMW) {
                //double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                //Players[i].Update(Game_data.Played_actions[t][i], Total_occupancies.back(), -noisy_loss, Capacities_t);
            }

            if (Players[i]->getType() == PlayerType::cGPMW) {
                //double noisy_loss = Game_data.Incurred_losses[t][i] + normal_distribution<double>(0, sigmas[i])(rng);
                //Players[i].Update_history(Game_data.Played_actions[t][i], -noisy_loss, Total_occupancies.back(), Capacities_t);
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
        //cout << Players[2].type << " run: " << run + 1 << ", time: " << t << ", Avg cong. " << fixed << setprecision(2) << avg_cong << endl;
        }
}

void Initialize_Players(int N, const std::vector<std::pair<int, int>>& od_Pairs, std::vector<std::vector<std::vector<int>>> &Strategy_vectors, std::vector<double> min_traveltimes, std::vector<double> max_traveltimes, std::vector<int> idxs_controlled, double T, std::string Algo, int version, std::vector<double> Sigma, std::vector<Eigen::MatrixXd>& Kernels, std::vector<double> sigmas, int numberofcontexts, std::vector<std::vector<double>> Capacities, std::vector<Player*>& Players) {
    for (int i = 0; i < N; i++) {
        int K_i = Strategy_vectors[i].size();
        double min_payoff = -max_traveltimes[i]; // min recompensa = - max tiempo viaje
        double max_payoff = -min_traveltimes[i];
        // cambiar la forma de tratar players
        // idxs son los ids de los agentes que son controlados
        if (find(idxs_controlled.begin(), idxs_controlled.end(), i) != idxs_controlled.end() && K_i > 1) { // si el agente está controlado por el agente y tiene más de un brazo
            if (Algo == "Hedge") {
                Players[i] = new Player_Hedge(K_i, T, min_payoff, max_payoff);
            }
            else if (Algo == "GPMW") {
                //Players[i] = new Player_GPMW(K_i, T, min_payoff, max_payoff, Strategy_vectors[i][0], Kernels[i][0], sigmas[i]);
            }
            else if (Algo == "cGPMW") {
                //Players[i] = new Player_cGPMW(K_i, T, min_payoff, max_payoff, Capacities[i][0], Strategy_vectors[i][0], Kernels[i][0], sigmas[i], version);
            }
        }
        else {
            K_i = 1;
            Players[i] = new Player_Hedge(K_i, T, min_payoff, max_payoff);
            for(int brazoborrado = 0;brazoborrado < 4; brazoborrado++)
                Strategy_vectors[i].pop_back();
            //eliminar 4 estrategias?
        }
        //Players[i]->OD_pair = OD_pairs[i]; 
        // ODPairs es una lista de pares donde cada i corresponde al agente
    }

}


std::vector<Eigen::MatrixXd> Optimize_Kernels(bool reoptimize, std::string Algo,  const std::vector<int>& idxs_controlled, const std::vector<std::vector<std::vector<int>>>& Strategy_vectors, const std::vector<double>& sigmas, int poly_degree, const std::vector<std::vector<double>>& Outcomes, const std::vector<std::vector<double>>& Capacities, const std::vector<std::vector<double>>& Payoffs, std::vector<std::vector<double>>& list_of_param_arrays)
{
    std::vector<Eigen::MatrixXd> Kernels(Strategy_vectors.size());

    // Cargar parametros para el algoritmo Algo
    std::string filename = "list_of_param_arrays_" + Algo + ".txt";
    list_of_param_arrays = loadParamsFromFile(filename);

    // Kernel tiene N jugadores pero solo se usan los indices de los jugadores controlados, el resto de los 500 jugadores estan vacios
    // Por lo tanto, para acceder a cada kernel hay que usar el indice de los jugadores controlados como la posición del kernel
    for (int jugador = 0; jugador < idxs_controlled.size(); jugador++) { // para cada jugador de los 20
        int ind = idxs_controlled[jugador];
        if (Kernels[ind].isZero()) { 

            std::vector<int> idx_nonzeros; // se usa en reoptimizacion -> entiendo que guarda las estrategias o brazos (de los 5 que hay) si no tienen valores de 0
            for (int carr = 0; carr < Strategy_vectors[ind][0].size(); carr++) { // para cada carretera se suman los valores de los caminos para ver si el jugador pasa por la carretera en algun camino
                int suma = 0;
                for (int camino = 0; camino < Strategy_vectors[ind].size(); ++camino) { 
                    suma += Strategy_vectors[ind][camino][carr];
                }
                if (suma != 0) idx_nonzeros.push_back(suma);
            }
            const int dim = idx_nonzeros.size();

            if (reoptimize == false) { // se hace en el init
                std::vector<double> loaded_params = list_of_param_arrays[ind]; // cargando parametros desde lista para Algo
                Eigen::VectorXd variances(2);
                variances << loaded_params[0], loaded_params[3];
                Eigen::VectorXd scales(2);
                scales << loaded_params[1], loaded_params[4];
                Eigen::VectorXd biases(2);
                biases << loaded_params[2], loaded_params[5];
                Eigen::MatrixXd active_dims(2, dim);
                for (int i = 0; i < dim; i++) {
                    active_dims(0, i) = i;
                    active_dims(1, i) = i + dim;
                }



                Eigen::MatrixXd kernel_1(dim, dim);
                Eigen::VectorXi active_dimsuno = active_dims.row(0).cast<int>().transpose();
                kernel_1 = poly_kernel(dim, 1, variances(0), scales(0), biases(0), active_dimsuno);

                Eigen::MatrixXd kernel_2(dim, dim);
                Eigen::VectorXi active_dimsdos = active_dims.row(1).cast<int>().transpose();
                kernel_2 = poly_kernel(dim, poly_degree, variances(1), scales(1), biases(1), active_dimsdos);

                Kernels[ind] = kernel_1.cwiseProduct(kernel_2);

            }
            else { // se hace en la ejecución del juego

            }
        }

    }
    return Kernels;
}


std::vector<std::vector<double>> loadParamsFromFile(std::string fileName)
{
    std::ifstream file(fileName);
    std::vector<std::vector<double>> params;

    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> paramLine;
            double val;
            while (iss >> val) {
                paramLine.push_back(val);
            }
            params.push_back(paramLine);
        }
        file.close();
    }
    else {
        std::cerr << "No se pudo abrir el archivo: " << fileName << std::endl;
    }

    return params;
}



Eigen::MatrixXd poly_kernel(int dim, int degree, double variance, double scale, double bias, const Eigen::VectorXi& active_dims) {
    int num_active_dims = active_dims.size();
    Eigen::MatrixXd kernel = Eigen::MatrixXd::Zero(dim, dim);

    // Convierte a tipo numérico común
    double pow_scale_var = scale * variance;
    double pow_bias = bias * bias;

    for (int i = 0; i < num_active_dims; ++i) {
        int d = active_dims(i);
        if (d < dim && d >= 0) {
            for (int j = 0; j <= degree; ++j) {
                for (int k = 0; k <= degree; ++k) {
                    // Realiza las operaciones con los mismos tipos
                    kernel(d, d) += std::pow(pow_scale_var, j + k) * std::pow(pow_bias, j) * std::pow(pow_bias, k);
                }
            }
        }
    }
    return kernel;
}





/*#include <vector>
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
}*/