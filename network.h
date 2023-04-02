#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

//includes para el algoritmo k_shortest_paths
#include <limits>
#include <set>
#include <map>
#include <queue>
//#include <string>
//#include <vector>
//#include <fstream>
//#include <iostream>
//#include <algorithm>

const int NUM_NODOS = 24;
const int K = 5; // max num routes

std::vector<std::vector<int>> read_csv_toInteger(std::string filename);
std::vector<std::vector<std::string>> read_csv(std::string filename);

typedef int CoordenadaIzq;
typedef int CoordenadaDer;
typedef int OD_Demand;


typedef struct 
{
    int numNodo;
    std::pair<CoordenadaIzq, CoordenadaDer> nodo;
} Nodo;

typedef struct
{
    int init_node;
    int term_node;
    double capacity;
    double lenght; // = weight TODO cambiar a int
    double freeFlowTime; // TODO cambiar a int
    int power;
} Carretera;

class NetworkData {
public: 
    NetworkData(std::vector<Nodo> n, std::vector<Carretera> c);
    std::vector<Nodo> getNodos() const;
    std::vector<Carretera> getCarreteras() const;
    int getNumCarreteras() const;
    int getNumNodos() const;

private:
    std::vector<Nodo> nodos;
    std::vector<Carretera> carreteras;
};

typedef struct
{
    /*int init_node;
    int term_node;
    float capacity;
    int lenght;
    int freeFlowTime;
    int power;*/
} Strategy;

//Create Network
NetworkData createNetwork();
std::vector<Nodo> crearListaNodos(const std::vector<std::vector<std::string>> & datosNodos);
std::vector<Carretera> crearListaCarreteras(const std::vector<std::vector<std::string>> & datosCarreteras);

// Compute Strategy Vectors
std::vector<std::vector<OD_Demand>> createOD_Demands();
void computeStrategyVectors(const NetworkData & network, const std::vector<std::vector<OD_Demand>> & od_Demands, int numRoutes, int multFactor);
int get_edge_idx(std::vector<Carretera>);
std::vector<std::vector<int>> k_shortest_paths(const NetworkData& network, const int& init_node, const int& term_node, const int& k_paths);



//TODO: Comentar o borrar al terminar la implementación
void printNodos(std::vector<Nodo> n);
void printCarreteras(std::vector<Carretera> c);
void printOD_Demands(std::vector<std::vector<OD_Demand>> d);