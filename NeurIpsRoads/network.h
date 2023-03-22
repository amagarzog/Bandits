#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

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
    // Los vectores en python eran matrices que había que hacerles reshape para poder trabajar con ellas commo si fueran vectores.
    // Entonces hay que ver en el "repeated_routing.py" como almacena los datos en matrices para almacenarlos, esta vez, en vectores. 
    NetworkData(std::vector<Nodo> n, std::vector<Carretera> c, std::vector<std::vector<int>> OD_Demands);

private:
    std::vector<Nodo> nodos;
    std::vector<Carretera> carreteras;
    std::vector<std::vector<int>> OD_Demands;
};


std::vector<std::vector<std::string>> read_csv(std::string filename);
NetworkData createNetwork();
std::vector<Nodo> crearListaNodos(std::vector<std::vector<std::string>> datosNodos);
std::vector<Carretera> crearListaCarreteras(std::vector<std::vector<std::string>> datosCarreteras);
std::vector<std::vector<int>> crearListaDemandas(std::vector<std::vector<std::string>> datosDemandas);
std::vector<std::vector<int>> computeStrategyVectors(std::vector<std::vector<int>>& OD_pares, NetworkData network);



// Aux
void printNodos(std::vector<Nodo> n);
void printCarreteras(std::vector<Carretera> c);