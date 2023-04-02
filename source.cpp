#include "network.h"

//Revisar TODO incluidos en el codigo
int main(...)
{
	NetworkData network = createNetwork();
	std::vector<std::vector<OD_Demand>> od_Demands = createOD_Demands(); //Es una matriz cuadrada que relaciona las demandas de un nodo i (fila) a un nodo j (columna)
	computeStrategyVectors(network, od_Demands, 5, 3);
}