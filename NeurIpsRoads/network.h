#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <queue>
#include <limits>


const int NUM_NODOS = 24;
const int K = 5; // max num routes

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

struct NodoConDistancia
{
    int nodo;
    int distancia;

    bool operator<(const NodoConDistancia& other) const
    {
        return distancia > other.distancia;
    }
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

std::vector<std::vector<int>> dijkstra(NetworkData network, int origen, int destino, int K);