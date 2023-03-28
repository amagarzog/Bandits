#ifndef network_h
#define network_h
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <queue>
#include <limits>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/graph_traits.hpp>


using namespace boost;

const int NUM_NODOS = 24;


typedef adjacency_list<vecS, vecS, directedS,
    no_property, property<edge_weight_t, int> > Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;

typedef graph_traits<Graph>::edge_descriptor Edge;


typedef std::pair<int, int> EdgeWeight;




typedef int NodoIzquierdo;
typedef int NodoDerecho;


typedef struct 
{
    int numNodo;
    std::pair<NodoIzquierdo, NodoDerecho> nodo;
} Nodo;

typedef struct
{
    int init_node;
    int term_node;
    float capacity;
    int lenght;
    int freeFlowTime;
    int power;
} Carretera;

class NetworkData {
public:
    NetworkData(std::vector<Nodo> n, std::vector<Carretera> c);

//private:
    std::vector<Nodo> nodos;
    std::vector<Carretera> carreteras;
};




std::vector<std::vector<std::string>> read_csv(std::string filename);
NetworkData createNetwork();
std::vector<std::vector<int>> takeDemands();
std::vector<Nodo> crearListaNodos(std::vector<std::vector<std::string>> datosNodos);
std::vector<Carretera> crearListaCarreteras(std::vector<std::vector<std::string>> datosCarreteras);
std::vector<std::vector<int>> crearListaDemandas(std::vector<std::vector<std::string>> datosDemandas);
std::vector<std::vector<int>> computeStrategyVectors(std::vector<std::vector<int>>& OD_Demands, NetworkData network);



// Aux
void printNodos(std::vector<Nodo> n);
void printCarreteras(std::vector<Carretera> c);

void dijkstra(const NetworkData& red, int nodoOrigen, std::vector<float>& distancias, std::vector<std::vector<int>>& caminos);
std::vector<std::vector<std::vector<int>>> k_shortest_paths(const NetworkData& red, int k);

#endif