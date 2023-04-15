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

    this->played_actions.push_back(played_action);
    for (int i = 0; i < played_actions.size(); i++) {
        //std::cout << "Brazo jugado Ronda " << i << ": " << played_actions[i] << ", ";
    }
    std::cout << std::endl;
    this->history_payoffs.push_back(payoff); 
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
    this->history.push_back(new_history);

    int beta_t = 0.5;
    std::vector<double> other_occupancies(idx_nonzeros.size()); // Las ocupaciones que no son por parte del jugador
    for (size_t i = 0; i < idx_nonzeros.size(); ++i) {
        other_occupancies[i] = total_occupancies[idx_nonzeros[i]] - strategy_vecs[played_action][idx_nonzeros[i]];
    }
    if (ronda > 0) {
        std::vector<sample_type> dlib_X_train = history_to_dlib_samples(this->history);
        std::vector<double> dlib_y_train = history_payoffs_to_dlib_labels(this->history_payoffs);

        double gamma = 0.25;
        //print_dlib_X_train(dlib_X_train, played_action, dlib_y_train);

        dlib::rvm_regression_trainer<kernel_type> trainer;
        trainer.set_kernel(kernel_type(gamma));
        dlib::decision_function<kernel_type> model = trainer.train(dlib_X_train, dlib_y_train);


        for (int a = 0; a < K_; a++) {
            dlib::matrix<double, 1, 0> x1(1, idx_nonzeros.size());
            for (int i = 0; i < idx_nonzeros.size(); ++i) {
                x1(0, i) = strategy_vecs[a][idx_nonzeros[i]];
            }

            dlib::matrix<double, 1, 0> x2(1, idx_nonzeros.size());
            for (int i = 0; i < idx_nonzeros.size(); ++i) {
                x2(0, i) = other_occupancies[i] + x1(0, i);
            }

            dlib::matrix<double, 1, 0> X_test(1, 2 * idx_nonzeros.size());
            set_subm(X_test, 0, 0, 1, idx_nonzeros.size()) = x1;
            set_subm(X_test, 0, idx_nonzeros.size(), 1, idx_nonzeros.size()) = x2;

            double prediction = model(X_test);
            double variance = calculate_residual_variance(model, dlib_X_train, dlib_y_train);
            //std::cout << a << " - Prediccion:   " << prediction << " " << variance << std::endl;


            this->ucb_rewards_est[a] = prediction + variance * beta_t;
            double payoff = std::max(this->ucb_rewards_est[a], min_payoff_);
            payoff = std::min(payoff, max_payoff_);

            double p1 = (payoff - min_payoff_);
            double p2 = (payoff - min_payoff_);
            double payoff_scaled = (payoff - min_payoff_) / (max_payoff_ - min_payoff_);
            double loss = 1.0 - payoff_scaled;
            this->cum_losses[a] = this->cum_losses[a] + loss;

            double weight = std::exp(-this->gamma_t_ * this->cum_losses[a]);
            this->weights_[a] = weight;
        }

        for (int j = 0; j < weights_.size(); j++)
            ;//std::cout << "Peso Brazo : "<< j << " - " << weights_[j] << " | ";
        //std::cout << std::endl;
    }
}

double calculate_residual_variance(dlib::decision_function<kernel_type>& model,
    const std::vector<sample_type>& dlib_X_train,
    const std::vector<double>& dlib_y_train) {

    double residuals_mean = 0.0;
    double residuals_variance = 0.0;
    for (size_t i = 0; i < dlib_X_train.size(); ++i) {
        double prediction = model(dlib_X_train[i]);
        double residual = dlib_y_train[i] - prediction;
        residuals_mean += residual;
        residuals_variance += residual * residual;
    }
    residuals_mean /= dlib_X_train.size();
    residuals_variance /= dlib_X_train.size() - dlib_X_train[0].size() - 1;

    return residuals_variance;
}



void print_dlib_X_train(const std::vector<sample_type>& dlib_X_train, int brazo, std::vector<double> payoffs) {
    for (std::size_t i = 0; i < dlib_X_train.size(); ++i) {
        std::cout << "Sample " << i + 1 << ":\n";
        for (std::size_t j = 0; j < dlib_X_train[i].size(); ++j) {
            std::cout << dlib_X_train[i](j) << " ";
        }
        std::cout  <<"Payoff: " << payoffs[i]<< std::endl;
    }
}

std::vector<sample_type> history_to_dlib_samples(const std::vector<std::vector<double>>& history) {
    std::vector<sample_type> dlib_samples;
    for (const auto& row : history) {
        sample_type sample(row.size());
        for (std::size_t i = 0; i < row.size(); ++i) {
            sample(i) = row[i];
        }
        dlib_samples.push_back(sample);
    }
    return dlib_samples;
}

std::vector<double> history_payoffs_to_dlib_labels(const std::vector<double>& history_payoffs) {
    return history_payoffs;
}

std::vector<double> Player_cGPMW::mixed_strategy()
{
    std::vector<double> strategy(K_);
    // strategy representa la probabilidad de elegir cada accion
    double sum_weights = accumulate(weights_.begin(), weights_.end(), 0.0);
    for (int i = 0; i < K_; i++) {
        strategy[i] = weights_[i] / sum_weights; // de esa forma se consigue que cada estrategia/brazo tenga un aprobabilidad de ser elegida entre 0 y 1 de forma ponderada con el resto de brazos
    }
    return strategy;
}

int Player_cGPMW::sample_action()
{
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