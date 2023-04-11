#include "algorithms.h"



std::vector<double> Player_Hedge::mixed_strategy() {
    std::vector<double> strategy(K_);
    // strategy representa la probabilidad de elegir cada accion
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        strategy[i] = weights_[i] / sum_weights; // de esa forma se consigue que cada estrategia/brazo tenga un aprobabilidad de ser elegida entre 0 y 1 de forma ponderada con el resto de brazos
    }
    return strategy;
}

int Player_Hedge::sample_action() {
    std::vector<double> strategy = mixed_strategy();
    double r = ((double)rand() / RAND_MAX); // de esta forma se consigue que r tenga un valor entre 0 y 1
    double sum_prob = 0.0;
    for (int i = 0; i < K_; i++) {
        sum_prob += strategy[i];
        if (r <= sum_prob) {
            return i; // se elige la acción
        }
    }
    return K_ - 1; // In case of numerical errors
}

void Player_Hedge::Update(std::vector<int> played_actions, int player_idx, const NetworkData &network, std::vector<double> Capacities_t, std::vector<std::vector<std::vector<int>>> Strategy_vectors) {
    std::vector<double> losses_hindsight(K_);
    for (int a = 0; a < K_; a++) {
        std::vector<int> modified_outcome = played_actions;
        modified_outcome[player_idx] = a; // modifica el brazo jugado a para ver cual seri ael travel time para los 5 brazos
        //perdidas si hubiera elegido el brazo a
        std::vector<double> traveltimetmp = Compute_traveltimes(network, Strategy_vectors, modified_outcome, player_idx, Capacities_t);
        losses_hindsight[a] = traveltimetmp[player_idx];
    }

    // Se establecen recompensas por cada accion y se limitan al maximo payoff y minimo payoff
    std::vector<double> payoffs(K_);
    for (int i = 0; i < K_; i++) {
        payoffs[i] = -losses_hindsight[i]; // las recompensas son los tiempos de viaje en negativo (cuanto menos tiempo de viaje, mejor recompensa)
        payoffs[i] = std::max(payoffs[i], min_payoff_);
        payoffs[i] = std::min(payoffs[i], max_payoff_); // delimitar entre min y maximo
    }

    // Se limitan los recompensas para cada brazo para que esten entre el intervalo 0 y 1
    std::vector<double> payoffs_scaled(K_);
    for (int i = 0; i < K_; i++) {
        payoffs_scaled[i] = (payoffs[i] - min_payoff_) / (max_payoff_ - min_payoff_); // - / - = +
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
    /*
    Si la pérdida es alta, entonces exp(-gamma_t_ * losses[i]) será un número pequeño y el nuevo peso w_i' será menor que el peso anterior w_i. Por otro lado, si la pérdida es baja o cero, 
    entonces exp(-gamma_t_ * losses[i]) será cercano a 1, lo que significa que el peso de la estrategia no cambia mucho en esa iteración. En general, el factor de actualización exponencial se 
    utiliza para ajustar la estrategia del jugador en función de su desempeño anterior, con el objetivo de mejorar su rendimiento en iteraciones futuras del juego.
    
    La tasa de aprendizaje hará que el peso se actualice con mayor o menor fuerza respecto a las pérdidas:
    Si la tasa de aprendizaje es alta las losses tendrán mas efecto en los weights ya que el exponencial sera un numero mayor.
    */

    // Se normalizan los pesos
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        weights_[i] = weights_[i] / sum_weights;
    }
}

std::vector<double> Player_GPMW::mixed_strategy() {
    std::vector<double> strategy(K_);
    // strategy representa la probabilidad de elegir cada accion
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        strategy[i] = weights_[i] / sum_weights; // de esa forma se consigue que cada estrategia/brazo tenga un aprobabilidad de ser elegida entre 0 y 1 de forma ponderada con el resto de brazos
    }
    return strategy;
}

int Player_GPMW::sample_action() {
    std::vector<double> strategy = mixed_strategy();
    double r = ((double)rand() / RAND_MAX); // de esta forma se consigue que r tenga un valor entre 0 y 1
    double sum_prob = 0.0;
    for (int i = 0; i < K_; i++) {
        sum_prob += strategy[i];
        if (r <= sum_prob) {
            return i; // se elige la acción
        }
    }
    return K_ - 1; // en caso de errores numericos
}


void Player_GPMW::Update(int ronda, int played_action, std::vector<double> total_occupancies, double payoff, std::vector<double> Capacities_t)
{

    //std::cout << "kernel tamaño " << kernel.size() << " " << this->idx_nonzeros.size();
    this->history_payoffs[ronda] = payoff; 
    std::vector<double> new_history;
    for (int i = 0; i < this->idx_nonzeros.size(); ++i) {
        int idx = this->idx_nonzeros[i];
        new_history.push_back(this->strategy_vecs[played_action][idx]);
        new_history.push_back(total_occupancies[idx]);
    }
    // en new history se ponen algunos ceros de strategyvecs porque estas cogiendo todas las carreteras qeu necesita el jugadore para los 5 brazos
    // con idx_nonzeros, pero al seleccionar un brazo jugado, habra estrategias que estaran usando la carretara mientras que el brazo jugado no, y como
    // se coge del brazo jugado, se mandaria un 0. Por ejemplo: del brazo 2 se usa la carretera 14 para mandar 3, pero del brazo 1 no, si se juega el brazo uno,
    // idx_nonzeros pasara por la carretera 14 y strategyvecs[1][14] = 0 porque lo usa el brazo 2
    this->history[ronda] =new_history;

    int beta_t = 0.5;

    std::vector<double> other_occupancies(idx_nonzeros.size()); // Las ocupaciones que no son por parte del jugador
    for (size_t i = 0; i < idx_nonzeros.size(); ++i) {
        other_occupancies[i] = total_occupancies[idx_nonzeros[i]] - strategy_vecs[played_action][idx_nonzeros[i]];
    }
    this->played_actions.push_back(played_action);

    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    if ((ronda) <= this->idx_nonzeros.size()) {
        X_train.resize(ronda, 2 * idx_nonzeros.size());
        for (int i = 0; i < ronda; ++i) {
            for (int j = 0; j < 2 * idx_nonzeros.size(); ++j) {
                X_train(i, j) = this->history[i][j];
            }
        }

        y_train.resize(ronda);
        for (int i = 0; i < ronda; ++i) {
            y_train(i) = this->history_payoffs[i];
        }
    }
    else { // se cogen kernel.size()=this->idx_nonzeros.size() datos aleatorios para que coincidan las matrices
        X_train.resize(this->idx_nonzeros.size(), 2 * idx_nonzeros.size());
        y_train.resize(this->idx_nonzeros.size());
        std::vector<int> indices(ronda);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Selecciona los primeros kernel.size() índices
        std::vector<int> kernel_indices(indices.begin(), indices.begin() + this->idx_nonzeros.size());
        for (int i = 0; i < this->idx_nonzeros.size(); ++i) {
            int idx = kernel_indices[i];
            for (int j = 0; j < 2 * idx_nonzeros.size(); ++j) {
                X_train(i, j) = this->history[idx][j];
            }
            y_train(i) = this->history_payoffs[idx];
        }

    }
    int noise2 = sigma_e * sigma_e;
    int noise = sigma_e;
    // Crear el modelo de regresión gaussiana
    GaussianProcessRegression gpr(kernel, noise);
    std::cout << kernel.size() << "X historia pasada::" << std::endl << X_train << std::endl << std::endl;
    std::cout << "Y Payoff pasada::" << std::endl << y_train << std::endl << std::endl;
    /*std::cout << kernel << std::endl; */

    // Entrenar el modelo
    gpr.train(X_train, y_train);

    std::vector<int> payoffs(K_);
    for (int a = 0; a < this->K_; a++) {
        Eigen::MatrixXd x1(1, idx_nonzeros.size());
        for (int i = 0; i < idx_nonzeros.size(); ++i) {
            x1(0, i) = strategy_vecs[a][idx_nonzeros[i]];
        }
        

        Eigen::MatrixXd x2(1, idx_nonzeros.size());
        for (int i = 0; i < idx_nonzeros.size(); ++i) {
            x2(0, i) = other_occupancies[i] + x1(0, i);
        }

        Eigen::MatrixXd X_test(1, 2 * idx_nonzeros.size());
        X_test << x1, x2;

        std::cout << "X tests" << X_test << std::endl;

        // Hacer predicciones
        //std::cout << X_test.cols() << " " << kernel.cols();

        std::pair<Eigen::VectorXd, Eigen::VectorXd> predictions_and_variances = gpr.predict(X_test);
        Eigen::VectorXd predictions = predictions_and_variances.first;
        Eigen::VectorXd variances = predictions_and_variances.second;

        // Imprimir resultados
        std::cout << "Brazo " << a << " Predictions:" << predictions << " Variance:" << variances << std::endl;
        
        this->ucb_rewards_est[a] = predictions(0) + beta_t * variances(0);
        this->mean_rewards_est[a] = predictions(0);
        this->std_rewards_est[a] = variances(0);
        payoffs.push_back(this->ucb_rewards_est[a]);
    }

    

    int wr = 34;

}


/*
def Update(self, played_action, total_occupancies, payoff, Capacities_t):

        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history = np.vstack((self.history, np.concatenate((self.strategy_vecs[played_action][self.idx_nonzeros].T, total_occupancies[self.idx_nonzeros].T), axis=1)))

        beta_t = 0.5

        m = GPy.models.GPRegression(self.history, self.history_payoffs, self.kernel)
        m.Gaussian_noise.fix(self.sigma_e ** 2)

        other_occupancies = total_occupancies[self.idx_nonzeros] - self.strategy_vecs[played_action][self.idx_nonzeros]
        for a1 in range(self.K):
            x1 = self.strategy_vecs[a1][self.idx_nonzeros]
            x2 = other_occupancies + x1
            mu, var = m.predict(np.concatenate((x1.T, x2.T), axis=1))
            sigma = np.sqrt(np.maximum(var, 1e-6))

            self.ucb_rewards_est[a1] = mu + beta_t * sigma
            self.mean_rewards_est[a1] = mu
            self.std_rewards_est[a1] = sigma

        payoffs = np.array(self.ucb_rewards_est)
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.cum_losses = self.cum_losses + losses

        gamma_t = self.gamma_t
        self.weights = np.exp(np.multiply(gamma_t, -self.cum_losses))
*/











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


/*VectorXr x1(this->idx_nonzeros.size());
      for (int i = 0; i < idx_nonzeros.size(); i++) {
          x1(i) = strategy_vecs[a][idx_nonzeros[i]];
      }
      VectorXr x2(x1.size());
      for (int i = 0; i < other_occupancies.size(); i++) { //misma size que idx_nonzeros
          x2(i) = other_occupancies[i] + x1(i); // total occupancies?
      }


      x_concatenated << x1, x2;
      x_concatenated.transposeInPlace();
      VectorXr x_in = Eigen::Map<VectorXr>(x_concatenated.data(), x_concatenated.size());*/

int Player::sample_action()
{
    return 0;
}

void Player::Update(std::vector<int> played_actions, int player_idx, const NetworkData& network, std::vector<double> Capacities_t, std::vector<std::vector<std::vector<int>>> Strategy_vectors)
{
}

void Player::Update(int ronda, int played_action, std::vector<double> total_occupancies, double payoff, std::vector<double> Capacities_t)
{
}

int Player::getK()
{
    return this->K_;
}

PlayerType Player::getType()
{
    return this->type_;
}
