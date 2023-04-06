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
#include <utility>
#include <limits>
#include <set>
#include <map>
#include <string>
#include "GraphElements.h"
#include "Graph.h"
#include "DijkstraShortestPathAlg.h"
#include "YenTopKShortestPathsAlg.h"

using namespace std;

const int NUM_NODOS = 24;




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
    int length;
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
void testYenAlg(int source, int dest, std::vector<Carretera> listaCarreteras);

// Aux
void printNodos(std::vector<Nodo> n);
void printCarreteras(std::vector<Carretera> c);


#endif