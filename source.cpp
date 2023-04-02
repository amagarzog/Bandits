/*
void print(std::vector<std::vector<std::string>> d) {
    for (auto fila : d) {
        for (auto campo : fila) {
            std::cout << campo << " ";
        }
        std::cout << std::endl;
    }
}
*/

#include <limits>
#include <set>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "GraphElements.h"
#include "Graph.h"
#include "DijkstraShortestPathAlg.h"
#include "YenTopKShortestPathsAlg.h"

#include "network.h"


void testDijkstraGraph() //TODO no hace falta
{
	Graph* my_graph_pt = new Graph("test_15");
	DijkstraShortestPathAlg shortest_path_alg(my_graph_pt);
	BasePath* result =
		shortest_path_alg.get_shortest_path(
			my_graph_pt->get_vertex(0), my_graph_pt->get_vertex(14));
	result->PrintOut(std::cout);
}

void testYenAlg() //TODO no hace falta
{
	Graph my_graph("test_15");

	YenTopKShortestPathsAlg yenAlg(my_graph, my_graph.get_vertex(0),
		my_graph.get_vertex(14));

	int i = 0;
	while (yenAlg.has_next())
	{
		++i;
		yenAlg.next()->PrintOut(std::cout);
	}
}

void testYenAlg(const NetworkData& network, const int& init_node, const int& term_node, const int& k_paths)
{
	Graph my_graph(network);

	YenTopKShortestPathsAlg yenAlg(my_graph, my_graph.get_vertex(init_node),
		my_graph.get_vertex(term_node));

	int i = 0;
	while (yenAlg.has_next() && i < k_paths)
	{
		++i;
		yenAlg.next()->PrintOut(std::cout);
	}
}

//Revisar TODO incluidos en el codigo
int main(...)
{
	NetworkData network = createNetwork();
	std::vector<std::vector<OD_Demand>> od_Demands = createOD_Demands(); //Es una matriz cuadrada que relaciona las demandas de un nodo i (fila) a un nodo j (columna)
	testYenAlg(network, 8, 13, 3);
}