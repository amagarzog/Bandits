#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

std::vector<std::vector<int>> read_csv_toInteger(std::string filename);
std::vector<std::vector<std::string>> read_csv(std::string filename);

typedef int CoordenadaIzq;
typedef int CoordenadaDer;
typedef int OD_Demand;
typedef std::pair<Nodo, Nodo> OD_Pair;


typedef struct 
{
    int numNodo;
    std::pair<CoordenadaIzq, CoordenadaDer> nodo;
} Nodo;

typedef std::pair<Nodo, Nodo> OD_Pair;

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

private:
    std::vector<Nodo> nodos;
    std::vector<Carretera> carreteras;
};

typedef struct
{
    int init_node;
    int term_node;
    float capacity;
    int lenght;
    int freeFlowTime;
    int power;
} Strategy;

//Create Network
NetworkData createNetwork();
std::vector<Nodo> crearListaNodos(const std::vector<std::vector<std::string>> & datosNodos);
std::vector<Carretera> crearListaCarreteras(const std::vector<std::vector<std::string>> & datosCarreteras);

// Compute Strategy Vectors
std::vector<std::vector<OD_Demand>> createOD_Demands();
void computeStrategyVectors(const NetworkData & network, const std::vector<std::vector<OD_Demand>> & od_Demands, int multFactor = 3);
int get_edge_idx(std::vector<Carretera>);



//TODO: Comentar o borrar al terminar la implementación
void printNodos(std::vector<Nodo> n);
void printCarreteras(std::vector<Carretera> c);
void printOD_Demands(std::vector<std::vector<OD_Demand>> d);