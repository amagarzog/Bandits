#include "simulation.h"


Simulation::Simulation(const NetworkData& network) : network(network) {
	this->network = network;
	this->od_Demands = createOD_Demands(); 
	this->Strategy_vectors = computeStrategyVectors(network, od_Demands, this->od_Pairs, num_routes, multFactor);
	this->numplayers = this->Strategy_vectors.size();
	selectParameters();
}

void Simulation::selectParameters(){
	/*Parámetros que se establecen para controlar el juego->por defecto toman estos valores*/
	this->controlledplayers = 20;
	this->rondas = 100;
	this->numcontextos = 10;
	this->polykernel = 4;
}

void Simulation::init(){
	// Índices de jugadores controlados, se obtienen aleatoriamente
	std::vector<int> aux(this->controlledplayers, 0);
	this->idcontrolledplayers = aux;
	std::vector<int> idxs_all(this->numplayers);
	std::iota(idxs_all.begin(), idxs_all.end(), 0);

	std::random_device rd;
	std::mt19937 g(rd()); 
	std::shuffle(idxs_all.begin(), idxs_all.end(), g);

	std::copy(idxs_all.begin(), idxs_all.begin() + this->controlledplayers, this->idcontrolledplayers.begin()); //elegir los controlledPlayers entre todos los players

	// Capacidades y contextos aleatorios
	std::vector<double> initCapacities = getCapacities(network);
	std::vector<std::vector<double>> Capacities;
	for (int c = 0; c < this->numcontextos; c++) {
		std::vector<double> perturbed_capacities;
		for (int i = 0; i < initCapacities.size(); i++) {
			double perturbation_factor = 0.7 + 0.5 * (double)std::rand() / RAND_MAX;
			double perturbed_capacity = perturbation_factor * initCapacities[i];
			perturbed_capacities.push_back(perturbed_capacity);
		}
		Capacities.push_back(perturbed_capacities); // unas capacidades para cada contexto
		// 76 capacidades (1 por carretera) distintas para cada contexto (10)
	}
	int runs = 1;
	std::srand(runs); // Intializes random number generator --> rand()
	std::vector<int> Contexts(this->rondas); // un contexto de los 10 posibles para cada ronda
	for (int i = 0; i < this->rondas; i++) {
		Contexts[i] = std::rand() % this->numcontextos; // el contexto de la ronda i es el resto de dividir un numero aleatorio entre el numero de contextos
	}



	std::vector<double> max_traveltimes(this->numplayers, 0.0); 
	std::vector<double> min_traveltimes(this->numplayers, 1e8);
	std::vector<std::vector<double>> Capacities_rand;
	std::vector<std::vector<double>> Outcomes;
	std::vector<std::vector<double>> Payoffs;
	for (int i = 0; i < this->rondas; i++) {
		std::vector<double> outcome(this->numplayers, 0.0); // todos los jugadores aplican la estrategia 0 por defecto
		for (int p : this->idcontrolledplayers) {
			int num_actions = Strategy_vectors[p].size();
			outcome[p] = std::rand() % num_actions; // el outcome va a ser la accion que juege el jugador: un numero entre 1 y 76 carreteras
		}
		std::vector<double> capacities = Capacities[std::rand() % this->numcontextos]; // se pasan las capacidades de un determinado contexto aleatorio
		std::vector<int> outcomeint(outcome.size());
		for (int i = 0; i < outcome.size(); i++) {
			outcomeint[i] = static_cast<int>(outcome[i]);
		}
	
		std::vector<double> traveltimes = Compute_traveltimes(this->network, Strategy_vectors, outcomeint, -1, capacities);
		for (int n = 0; n < this->numplayers; n++) {
			max_traveltimes[n] = std::max(max_traveltimes[n], traveltimes[n] + 0.01);
			min_traveltimes[n] = std::min(min_traveltimes[n], traveltimes[n] - 0.01);
		}
		Capacities_rand.push_back(capacities);
		Outcomes.push_back(outcome);
		std::vector<double> payoffs(this->numplayers);
		for (int n = 0; n < this->numplayers; n++) {
			payoffs[n] = -traveltimes[n];
		}
		Payoffs.push_back(payoffs);
	}

	/*
	 M = 100
    max_traveltimes = np.zeros(N)
    min_traveltimes = 1e8 * np.ones(N)
    Capacities_rand = []
    Outcomes = []
    Payoffs = []
    for i in range(M):
        outcome = np.zeros(N)  # all play first action by default
        for p in idxs_controlled:
            outcome[p] = np.random.randint(len(Strategy_vectors[p]))
        capacities =  np.array(Capacities[np.random.randint(0, C)])
        traveltimes = Compute_traveltimes(SiouxNetwork_data, Strategy_vectors, outcome.astype(int), 'all', capacities)
        max_traveltimes = np.maximum(max_traveltimes, traveltimes + 0.01)
        min_traveltimes = np.minimum(min_traveltimes, traveltimes - 0.01)
        Capacities_rand.append(capacities)
        Outcomes.append(outcome)
        Payoffs.append(-traveltimes)
	*/



}



void Simulation::run()
{
}


