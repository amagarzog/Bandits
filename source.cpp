#pragma once
#include "simulation.h"
#include "network.h"


//Revisar TODO incluidos en el codigo
int main(...)
{
	NetworkData network = createNetwork();
	Simulation simulation (network);
	simulation.init();
	simulation.run();


	return 0;
}