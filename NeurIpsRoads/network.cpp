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

NetworkData::NetworkData(std::vector<Nodo> n, std::vector<Carretera> c, std::vector<std::vector<int>> OD_Demands) {
    this->nodos = n;
    this->carreteras = c;
    this->OD_Demands = OD_Demands;
}

NetworkData createNetwork() {
    std::vector<std::vector<std::string>> datosNodos = read_csv("SiouxFalls_node.csv");
    std::vector<Nodo> listaNodos = crearListaNodos(datosNodos);
    std::vector<std::vector<std::string>> datosCarreteras = read_csv("SiouxFalls_net.csv");
    std::vector<Carretera> listaCarreteras = crearListaCarreteras(datosCarreteras);
    std::vector<std::vector<std::string>> datosDemandas = read_csv("SiouxFalls_OD_matrix.txt");
    std::vector<std::vector<int>> OD_Demands = crearListaDemandas(datosDemandas);

    NetworkData network(listaNodos, listaCarreteras, OD_Demands);
    return network;

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
    std::cout << "hola";
    return OD_Demands;
}

std::vector<std::vector<int>> computeStrategyVectors(std::vector<std::vector<int>>& OD_pares, NetworkData network){
    std::vector<std::vector<int>> strategyV;


    return strategyV;
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