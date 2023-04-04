#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include <random> 
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include "network.h"

const int num_routes = 5;
const int multFactor = 1;


class Simulation {

public:
	Simulation(const NetworkData& network);
	void init();
	void run();

private:
	//players
	int numplayers, controlledplayers;
	std::vector<int> idcontrolledplayers;

	//network
	NetworkData network;
	std::vector<std::vector<OD_Demand>> od_Demands;
	std::vector<std::vector<std::vector<int>>> Strategy_vectors;
	std::vector<std::pair<int, int>> od_Pairs;

	//parametros simulacion
	int rondas, polykernel, numcontextos;


	void selectParameters();

};