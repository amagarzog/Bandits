#include "network.h"

//includes para el algoritmo k_shortest_paths

#include "GraphElements.h"
#include "Graph.h"
#include "DijkstraShortestPathAlg.h"
#include "YenTopKShortestPathsAlg.h"

std::vector<std::vector<int>> read_csv_toInteger(std::string filename) {
    std::ifstream archivo_csv(filename);
    if (!archivo_csv)
    {
        std::cerr << "The file " << filename << " can not be opened!" << std::endl;
        exit(1);
    }

    std::vector<std::vector<int>> datos;

    std::string linea;

    while (std::getline(archivo_csv, linea)) {
        std::vector<int> fila;
        std::stringstream ss(linea);
        std::string campo;

        while (std::getline(ss, campo, ',')) {
            fila.push_back(std::stoi(campo));
        }

        datos.push_back(fila);
    }

    archivo_csv.close();

    return datos;
}

std::vector<std::vector<std::string>> read_csv(std::string filename) {
    std::ifstream archivo_csv(filename);
    if (!archivo_csv)
    {
        std::cerr << "The file " << filename << " can not be opened!" << std::endl;
        exit(1);
    }

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

std::vector<Nodo> NetworkData::getNodos() const {
    return this->nodos;
}
std::vector<Carretera> NetworkData::getCarreteras() const {
    return this->carreteras;
}

int NetworkData::getNumCarreteras() const {
    return this->carreteras.size();
}
int NetworkData::getNumNodos() const {
    return this->nodos.size();
}


NetworkData createNetwork() {
    std::vector<std::vector<std::string>> datosNodos = read_csv("SiouxFalls_node.csv"); //TODO cambiar archivo
    std::vector<Nodo> listaNodos = crearListaNodos(datosNodos);
    std::vector<std::vector<std::string>> datosCarreteras = read_csv("SiouxFalls_net.csv");//TODO cambiar archivo
    std::vector<Carretera> listaCarreteras = crearListaCarreteras(datosCarreteras);
    NetworkData network(listaNodos, listaCarreteras);
    return network;
}

std::vector<Nodo> crearListaNodos(const std::vector<std::vector<std::string>> & datosNodos) {
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

std::vector<Carretera> crearListaCarreteras(const std::vector<std::vector<std::string>> & datosCarreteras) {
    std::vector<Carretera> listaCarreteras;
    for (int i = 1; i < datosCarreteras.size(); i++) {
        std::vector<std::string> fila = datosCarreteras[i];
        Carretera c;
        c.init_node = std::stoi(fila[0]);
        c.term_node = std::stoi(fila[1]);
        c.capacity = std::stod(fila[2]) / 100;
        c.lenght = std::stod(fila[3]);
        c.freeFlowTime = std::stod(fila[4]);
        c.power = std::stoi(fila[6]);
        listaCarreteras.push_back(c);
    }
    //printCarreteras(listaCarreteras);
    return listaCarreteras;
}

int get_edge_idx(std::vector<Carretera>) {
    int id=0;
    return id;
}

std::vector<std::vector<OD_Demand>> createOD_Demands() {
    std::vector<std::vector<OD_Demand>> od_Demands = read_csv_toInteger("SiouxFalls_OD_matrix.txt");
    //printOD_Demands(od_Demands);
    return od_Demands;
}

int getidx(const NetworkData& network, int nodo1, int nodo2) {
    int idx = 0;
    std::vector<Carretera> carrs = network.getCarreteras();
    for (int i = 0; i < carrs.size(); i++) {
        if (carrs[i].init_node == nodo1 && carrs[i].term_node == nodo2) {
            idx = i;
        }
    }
    return idx;
}


void computeStrategyVectors(const NetworkData& network, const std::vector<std::vector<OD_Demand>>& od_Demands,  int numRoutes, int multFactor) {
    std::vector<std::pair<int, int>> od_Pairs;
    std::vector<int> demands;

    for (int i = 0; i < NUM_NODOS; i++) {
        for (int j = 0; j < NUM_NODOS; j++) {
            if (od_Demands[i][j] > 0) {
                od_Pairs.push_back({ i + 1, j + 1 });
                demands.push_back(od_Demands[i][j] / 100);
            }
        }
    }
    
    int E = network.getNumCarreteras(); //esto hace que no pueda poner const en el param network
    std::vector<std::vector<std::vector<int>>> paths(od_Pairs.size()); 
    for (int i = 0; i < od_Pairs.size(); i++) {
        std::vector<std::vector<int>> k_shortest_paths_between_od_pair = k_shortest_paths(network, od_Pairs[i].first, od_Pairs[i].second, numRoutes);
        paths[i] = k_shortest_paths_between_od_pair;
    }
    std::vector<std::vector<int>> Strategy_vectors;
    for (int a = 0; a < paths.size(); ++a) {
        std::vector<int> vec(E, 0);
        std::vector<std::vector<int>> pathtmp = paths[a];
        for (int n = 0; n < pathtmp.size(); ++n) { // 5 caminos
            std::vector<int> kpath = pathtmp[n];
            for (int d = 0; d < kpath.size()-1; d++) { // nodos de cada camino (de los 5)
                int idx = getidx(network, kpath[d], kpath[d+1]);
                vec[idx] = 1; // se marca la carretera entre los dos nodos
            }
        }
        std::vector<int> strategyvec(E, 0);
        std::vector<Carretera> carreteras = network.getCarreteras();
        std::vector<int> Freeflowtimes(E);
        for (int j = 0; j < E; j++) { //  TODO MEJORAR RENDIMIENTO (unordered map odpairs)
            if (vec[j] == 1) {
                bool done = false;
                for (int i = 0; i < od_Pairs.size() && !done; i++) {
                    if (od_Pairs[i].first == carreteras[j].init_node && od_Pairs[i].second == carreteras[j].term_node) {
                        done = true;
                        strategyvec[j] = demands[i];
                    }
                }

            }
            Freeflowtimes[j] = carreteras[j].freeFlowTime;
        }

        if (a == 0) {
            Strategy_vectors.push_back(strategyvec);
        }
        else if (dot_product(strategyvec, Freeflowtimes) < 1*dot_product(Strategy_vectors[0], Freeflowtimes)) {
            Strategy_vectors.push_back(strategyvec);
        }

        // return strategyvectos
    }

    int dot_product(std::vector<int> vec1, std::vector<int> vec2) {
        int result = 0.0;
        for (int i = 0; i < vec1.size(); i++) {
            result += vec1[i] * vec2[i];
        }
        return result;
    }


    /*int E = Edges.size();
    int K = num_routes;
    vector<vector<vector<double>>> Strategy_vectors(OD_pairs.size());
    for (int i = 0; i < OD_pairs.size(); ++i) {
        Strategy_vectors[i].resize(K);
        Matrix OD_pair = { {OD_pairs[i][0]}, {OD_pairs[i][1]} };
        vector<vector<string>> paths = k_shortest_paths(Networkx, to_string(OD_pair[0][0]), to_string(OD_pair[1][0]), K, "weight");
        for (int a = 0; a < paths.size(); ++a) {
            vector<double> vec(E, 0);
            for (int n = 0; n < paths[a].size() - 1; ++n) {
                int idx = get_edge_idx(Edges, stoi(paths[a][n]), stoi(paths[a][n + 1]));
                vec[idx] = 1;
            }
            vector<double> strategy_vec(E);
            transform(vec.begin(), vec.end(), strategy_vec.begin(), [&](double x) { return x * Demands[i]; });
            if (a == 0) {
                Strategy_vectors[i][0] = strategy_vec;
            }
            else if (inner_product(strategy_vec.begin(), strategy_vec.end(), Freeflowtimes[0].begin(), 0.0) < multipl_factor * inner_product(Strategy_vectors[i][0].begin(), Strategy_vectors[i][0].end(), Freeflowtimes[0].begin(), 0.0)) {
                Strategy_vectors[i][a] = strategy_vec;
            }
        }
    }*/


    //return strategyVector;
        
}

std::vector<std::vector<int>> k_shortest_paths(const NetworkData& network, const int& init_node, const int& term_node, const int& k_paths)
{
    Graph my_graph(network);

    YenTopKShortestPathsAlg yenAlg(my_graph, my_graph.get_vertex(init_node),
        my_graph.get_vertex(term_node));


    int i = 0;
    std::vector<std::vector<int>> paths;
    while (yenAlg.has_next() && i < k_paths)
    {
        ++i;
        yenAlg.next()->Report(paths);
    }

    return paths;
}



/*def Compute_Strategy_vectors(OD_demands, Freeflowtimes, Networkx, Edges, num_routes = 5, mult_factor = None):
    if mult_factor is None:
        multipl_factor = 3
    else:
        multipl_factor = mult_factor
    OD_pairs = []
    Demands = []
    for i in range(24):
        for j in range(24):
            if OD_demands[i, j] > 0:
                OD_pairs.append([i + 1, j + 1])
                Demands.append(OD_demands[i, j] / 100)

    E = len(Edges)
    K = num_routes  # K shortest paths for each agent
    Strategy_vectors = [()]*len(OD_pairs)
    for i in range(len(OD_pairs)):
        Strategy_vectors[i] = list()
        OD_pair = np.array(OD_pairs[i])
        paths = k_shortest_paths(Networkx, str(OD_pair[0]), str(OD_pair[1]), K, weight = 'weight')
        TODO
        for a in range(len(paths)):
            vec = np.zeros((E,1))
            for n in range(len(paths[a])-1):
                idx = get_edge_idx(Edges,  paths[a][n], paths[a][n+1])
                vec[idx] = 1
            strategy_vec = np.multiply(vec, Demands[i])
            if a == 0:
                Strategy_vectors[i].append(  strategy_vec )
            if a > 0 and np.dot(strategy_vec.T, Freeflowtimes) < multipl_factor* np.dot(Strategy_vectors[i][0].T, Freeflowtimes ):
                Strategy_vectors[i].append(strategy_vec )

    return Strategy_vectors, OD_pairs
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

void printOD_Demands(std::vector<std::vector<OD_Demand>> d) {
    for (auto nodo : d) {
        for (auto demanda : nodo) {
            std::cout << demanda << " ";
        }        
        std::cout << std::endl;
    }
}