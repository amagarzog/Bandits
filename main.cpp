#pragma once
#include "simulation.h"
#include "network.h"

int main(...)
{
	NetworkData network = createNetwork();
	Simulation simulation(network);
	simulation.init();
	simulation.run();



	return 0;
}