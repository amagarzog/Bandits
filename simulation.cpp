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

	std::copy(idxs_all.begin(), idxs_all.begin() + this->controlledplayers, this->idcontrolledplayers.begin());

	// Capacidades y contextos aleatorios
	std::vector<int> initCapacities = getCapacities(network);
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
	std::srand(runs);
	std::vector<int> Contexts(this->rondas); // un contexto de los 10 posibles para cada ronda
	for (int i = 0; i < this->rondas; i++) {
		Contexts[i] = std::rand() % this->numcontextos;
	}




}



void Simulation::run()
{
}


