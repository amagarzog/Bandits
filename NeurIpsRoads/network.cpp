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
    Graph G;
    for (const auto& nodo : network.nodos) {
        //G.add_node(nodo.numNodo);
    }

    for (const auto& carretera : network.carreteras) {
        //G.add_edge(carretera.init_node, carretera.term_node);
        // Puedes agregar los atributos de la carretera como atributos del borde (edge) en el grafo.
        // Por ejemplo, si la carretera tiene una capacidad, puedes hacer lo siguiente:
        //G[{carretera.init_node, carretera.term_node}]["capacity"] = carretera.capacity;
        // Repite esto para cada atributo que quieras agregar al borde.
    }

    
    
    std::vector<std::vector<std::vector<int>>> paths = k_shortest_paths(network, 5);





    return strategyV;
}


std::vector<std::vector<int>> k_shortest_paths(boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int> >& G, vertex_descriptor source, vertex_descriptor target, int k) {
    std::vector<std::vector<int>> result;

    // calculate the shortest path from source to target
    std::vector<boost::vertex_descriptor> path;
    boost::dijkstra_shortest_paths(G, source,
        boost::predecessor_map(boost::make_iterator_property_map(
            path.begin(), boost::get(boost::vertex_index, G))));

    // initialize a priority queue to store candidate paths
    std::priority_queue<std::vector<int>, std::vector<std::vector<int>>,
        std::greater<std::vector<int>>> candidates;

    // add the shortest path as the first candidate
    std::vector<int> shortest_path;
    for (int i = 0; i < path.size(); i++) {
        shortest_path.push_back(path[i]);
    }
    candidates.push(shortest_path);

    // iterate until kth shortest path is found or there are no more candidates
    while (!candidates.empty() && result.size() < k) {
        std::vector<int> candidate = candidates.top();
        candidates.pop();

        // check if the candidate path reaches the target
        if (candidate.back() == target) {
            result.push_back(candidate);
        }

        // generate new candidates by removing edges from the candidate path
        BOOST_FOREACH(boost::graph_traits<boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int> > >::edge_descriptor edge, boost::edges(G)) {
            vertex_descriptor u = boost::source(edge, G);
            vertex_descriptor v = boost::target(edge, G);
            if (candidate.size() >= 2 && candidate[candidate.size() - 2] == u
                && candidate.back() == v) {
                // remove the edge and add the new candidate
                std::vector<int> new_candidate = candidate;
                new_candidate.pop_back();
                new_candidate.back() = u;
                boost::dijkstra_shortest_paths(G, u,
                    boost::predecessor_map(boost::make_iterator_property_map(
                        new_candidate.begin(), boost::get(boost::vertex_index, G))));
                candidates.push(new_candidate);
            }
        }
    }

    return result;
}











/*
std::vector<std::vector<int>> dijkstra(NetworkData network, int origen, int destino, int K) {
    std::vector<std::vector<int>> caminos;
  
   /* std::priority_queue<NodoConDistancia> cola;
    std::vector<int> distancias(network.nodos.size(), std::numeric_limits<int>::max());
    std::vector<bool> visitados(network.nodos.size(), false);
    std::vector<std::vector<int>> antecesores(network.nodos.size());
    std::vector<int> auxpos(network.nodos.size(), 0);


    cola.push({ origen, 0 });
    distancias[origen - 1] = 0;

    while (!cola.empty()) {
        NodoConDistancia actual = cola.top();
        cola.pop();
        std::cout << actual.nodo << std::endl;


        if (actual.nodo == 5) {// || actual.nodo == 11 || actual.nodo == 3) {
            int a;
            a = 23;
        }

        if (actual.nodo == destino && caminos.size() < K) {
            std::vector<int> camino;
            int nodoActual = destino;
            while (nodoActual != origen) {
                camino.push_back(nodoActual);
                int ind = auxpos[nodoActual - 1];
                nodoActual = antecesores[nodoActual - 1][ind];
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
                int nuevaDistancia = distancias[actual.nodo - 1] + carretera.lenght;
                if (nuevaDistancia < distancias[nodoVecino - 1]) {
                    distancias[nodoVecino - 1] = nuevaDistancia;
                    cola.push({ nodoVecino, nuevaDistancia });
                    //antecesores[nodoVecino-1].clear();
                    antecesores[nodoVecino - 1].push_back(actual.nodo);
                    if (destino == nodoVecino) {
                        auxpos[nodoVecino - 1] = antecesores[nodoVecino - 1].size() - 1;
                    }
                }
                else if (nuevaDistancia == distancias[nodoVecino - 1]) {
                    antecesores[nodoVecino - 1].push_back(actual.nodo);
                }
            }
        }

    }
    
    return caminos;
}

*/


















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