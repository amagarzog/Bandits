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
    for (int i = 1; i < datosCarreteras.size(); i++) {
        std::vector<std::string> fila = datosCarreteras[i];
        Carretera c;
        c.init_node = std::stoi(fila[0]);
        c.term_node = std::stoi(fila[1]);
        c.capacity = std::stof(fila[2]) / 100;
        c.lenght = std::stoi(fila[3]);
        c.freeFlowTime = std::stoi(fila[4]);
        c.power = std::stoi(fila[6]);
        listaCarreteras.push_back(c);
    }
    //printCarreteras(listaCarreteras);
    return listaCarreteras;
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
    for (int i = 0; i < OD_Pair.size(); i++) {
        //std::cout << OD_Pair[i].first << OD_Pair[i].second << std::endl;
    }
    std::vector<std::vector<int>> path = dijkstra(network, 1, 4, 5);




    return strategyV;
}



std::vector<std::vector<int>> dijkstra(NetworkData network, int origen, int destino, int K) {
    std::vector<std::vector<int>> caminos;
    std::priority_queue<NodoConDistancia> cola;
    std::vector<int> distancias(network.nodos.size(), std::numeric_limits<int>::max());
    std::vector<bool> visitados(network.nodos.size(), false);
    std::vector<std::vector<int>> antecesores(network.nodos.size());
    std::vector<int> auxpos(network.nodos.size(), 0);


    cola.push({ origen, 0 });
    distancias[origen-1] = 0;

    while (!cola.empty()) {
        NodoConDistancia actual = cola.top();
        cola.pop();
        std::cout << actual.nodo << std::endl;
        
        
        if (actual.nodo == 5){// || actual.nodo == 11 || actual.nodo == 3) {
            int a;
            a = 23;
        }

        if (actual.nodo == destino && caminos.size() < K) {
            std::vector<int> camino;
            int nodoActual = destino;
            while (nodoActual != origen) {
                camino.push_back(nodoActual);
                int ind = auxpos[nodoActual - 1];
                nodoActual = antecesores[nodoActual-1][ind];
            }
            camino.push_back(origen);
            //reverse(camino.begin(), camino.end());
            caminos.push_back(camino);
            if (caminos.size() == K) {
                break;
            }
        }

        for (int i = 0; i < network.carreteras.size(); i++) {
            Carretera carretera = network.carreteras[i];
            if (carretera.init_node == actual.nodo) {
                int nodoVecino = carretera.term_node;
                int nuevaDistancia = distancias[actual.nodo-1] + carretera.lenght;
                if (nuevaDistancia < distancias[nodoVecino-1]) {
                    distancias[nodoVecino-1] = nuevaDistancia;
                    cola.push({ nodoVecino, nuevaDistancia });
                    //antecesores[nodoVecino-1].clear();
                    antecesores[nodoVecino-1].push_back(actual.nodo);
                    if (destino == nodoVecino) {
                        auxpos[nodoVecino - 1] = antecesores[nodoVecino - 1].size() - 1;
                    }
                }
                else if (nuevaDistancia == distancias[nodoVecino-1]) {
                    antecesores[nodoVecino-1].push_back(actual.nodo);
                }
            }
        }

    }

    return caminos;
}


void printNodos(std::vector<Nodo> n) {
    for (auto nodo : n) {
        std::cout << nodo.numNodo << " " << nodo.nodo.first << " " << nodo.nodo.second << std::endl;
    }
}

void printCarreteras(std::vector<Carretera> c) {
    for (auto carretera : c) {
        std::cout << carretera.init_node << " " << carretera.term_node << " " << carretera.capacity << " " << carretera.lenght << " " << carretera.freeFlowTime << " " << carretera.power << std::endl;
    }
}