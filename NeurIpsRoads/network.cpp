#include "network.h"

std::vector<std::vector<std::string>> read_csv(std::string filename) {
    std::ifstream archivo_csv(filename);

    std::vector<std::vector<std::string>> datos;

    std::string linea;

    while (std::getline(archivo_csv, linea)) {
        std::vector<std::string> fila;
        std::stringstream ss(linea);
        std::string campo;

        while (std::getline(ss, campo, ',')) {
            fila.push_back(campo);
        }

        datos.push_back(fila);
    }

    archivo_csv.close();

    return datos;
}

NetworkData::NetworkData(std::vector<Nodo> n, std::vector<Carretera> c) {
    this->nodos = n;
    this->carreteras = c;
}

NetworkData createNetwork() {
    std::vector<std::vector<std::string>> datosNodos = read_csv("SiouxFalls_node.csv");
    std::vector<Nodo> listaNodos = crearListaNodos(datosNodos);
    std::vector<std::vector<std::string>> datosCarreteras = read_csv("SiouxFalls_net.csv");
    std::vector<Carretera> listaCarreteras = crearListaCarreteras(datosCarreteras);

    NetworkData network(listaNodos, listaCarreteras);
    return network;
}

std::vector<std::vector<int>> takeDemands() {
    std::vector<std::vector<std::string>> datosDemandas = read_csv("SiouxFalls_OD_matrix.txt");
    std::vector<std::vector<int>> OD_Demands = crearListaDemandas(datosDemandas);
    return OD_Demands;
}


std::vector<Nodo> crearListaNodos(std::vector<std::vector<std::string>> datosNodos) {
    std::vector<Nodo> listaNodos;
    Nodo n;
    listaNodos.push_back(n);
    for (int i = 1; i < datosNodos.size(); i++) {
        std::vector<std::string> fila = datosNodos[i];
        Nodo n;
        n.numNodo = std::stoi(fila[0]);
        n.nodo.first = std::stoi(fila[1]);
        n.nodo.second = std::stoi(fila[2]);
        listaNodos.push_back(n);
    }
    //printNodos(listaNodos);
    return listaNodos;
}

std::vector<Carretera> crearListaCarreteras(std::vector<std::vector<std::string>> datosCarreteras) {
    std::vector<Carretera> listaCarreteras;
    Carretera a;
    listaCarreteras.push_back(a);
    for (int i = 1; i < datosCarreteras.size(); i++) {
        std::vector<std::string> fila = datosCarreteras[i];
        Carretera c;
        c.init_node = std::stoi(fila[0]);
        c.term_node = std::stoi(fila[1]);
        c.capacity = std::stof(fila[2]) / 100;
        c.length = std::stoi(fila[3]);
        c.freeFlowTime = std::stoi(fila[4]);
        c.power = std::stoi(fila[6]);
        listaCarreteras.push_back(c);
    }
    //printCarreteras(listaCarreteras);
    return listaCarreteras;
}


void printNodos(std::vector<Nodo> n) {
    for (auto nodo : n) {
        std::cout << nodo.numNodo << " " << nodo.nodo.first << " " << nodo.nodo.second << std::endl;
    }
}

void printCarreteras(std::vector<Carretera> c) {
    for (auto carretera : c) {
        std::cout << carretera.init_node << " " << carretera.term_node << " " << carretera.capacity << " " << carretera.length << " " << carretera.freeFlowTime << " " << carretera.power << std::endl;
    }
}

std::vector<std::vector<int>> crearListaDemandas(std::vector<std::vector<std::string>> datosDemandas) {
    std::vector<std::vector<int>> OD_Demands;
    for (int i = 1; i < datosDemandas.size(); i++) {
        std::vector<std::string> fila = datosDemandas[i];
        std::vector<int> aux;
        for (int j = 0; j < datosDemandas[i].size(); j++) {
            aux.push_back(std::stoi(datosDemandas[i][j]));
        }
        OD_Demands.push_back(aux);
    }
    return OD_Demands;
}

std::vector<std::vector<int>> computeStrategyVectors(std::vector<std::vector<int>>& OD_Demands, NetworkData network){
    std::vector<std::pair<int, int>> OD_Pair;
    std::vector<int> Demands;

    for (int i = 0; i < NUM_NODOS; i++) {
        for (int j = 0; j < NUM_NODOS; j++) {
            if (OD_Demands[i][j] > 0) {
                OD_Pair.push_back({ i + 1, j + 1 });
                Demands.push_back(OD_Demands[i][j] / 100);
            }
        }
    }

    std::vector<std::vector<int>> strategyV (OD_Pair.size());
 


    return strategyV;
}



void testYenAlg(int source, int dest, std::vector<Carretera> listaCarreteras)
{
   // Graph my_graph(listaCarreteras);
    /*
    YenTopKShortestPathsAlg yenAlg(my_graph, my_graph.get_vertex(source), my_graph.get_vertex(dest));

    std::vector<std::pair<std::vector<int>, double>> paths;

    int k = 5, cont = 0;
    while (yenAlg.has_next() && cont < k)
    {
        BasePath* p = yenAlg.next();
        std::pair<std::vector<int>, double> path = p->getPath(p);

        paths.push_back(path);
        cont++;
    }

    // Imprimir los caminos
    for (int i = 0; i < paths.size(); i++) {
        std::cout << "Camino " << i + 1 << " - Costo: " << paths[i].second << " - ";
        for (int j = 0; j < paths[i].first.size(); j++) {
            std::cout << paths[i].first[j] << " ";
        }
        std::cout << std::endl;
    }*/
}


