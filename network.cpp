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


std::vector<std::vector<std::vector<int>>> computeStrategyVectors(const NetworkData& network, std::vector<std::vector<OD_Demand>>& od_Demands, std::vector<std::pair<int, int>>& od_Pairs, int numRoutes, int multFactor) {
    std::vector<int> demands;

    for (int i = 0; i < NUM_NODOS; i++) {
        for (int j = 0; j < NUM_NODOS; j++) {
            if (od_Demands[i][j] > 0) {
                od_Pairs.push_back({ i + 1, j + 1 });
                demands.push_back(od_Demands[i][j] / 100);
            }
        }
    }

    std::vector<int> Freeflowtimes = getFreeFlowTimes(network);

    int E = network.getNumCarreteras(); //esto hace que no pueda poner const en el param network

    std::vector<std::vector<std::vector<int>>> Strategy_vectors; // tamaño 528 = 24*24

    //Para cada odPair se tiene k = 5 caminos posibles. Cada camino es un vector de carreteras.
    std::vector<std::vector<std::vector<int>>> paths(od_Pairs.size()); // tamaño = nº de odPAirs


    for (int i = 0; i < od_Pairs.size(); i++) {
        std::vector<std::vector<int>> Strategy_vector;
        std::vector<std::vector<int>> k_shortest_paths_between_od_pair = k_shortest_paths(network, od_Pairs[i].first, od_Pairs[i].second, numRoutes);
        paths[i] = k_shortest_paths_between_od_pair; // Para cada odPair se tiene k = 5 caminos posibles

        for (int camino = 0; camino < paths[i].size(); camino++) {
            std::vector<int> vec(E, 0);
            for (int nodo = 0; nodo < paths[i][camino].size() - 1; nodo++) { // nodos de cada camino (de los 5)
                int idx = getidx(network, paths[i][camino][nodo], paths[i][camino][nodo + 1]);
                vec[idx] = 1; // se marca la carretera entre los dos nodos
            }
            std::vector<int> strategyvec(E, 0); //Para cada carretera (76) almaceno la demanda de cada jugador en esa carretera
            std::vector<Carretera> carreteras = network.getCarreteras();
            for (int j = 0; j < E; j++) { //  TODO MEJORAR RENDIMIENTO (unordered map odpairs)
                if (vec[j] == 1) {
                    bool done = false;
                    for (int i = 0; i < od_Pairs.size() && !done; i++) {
                        if (od_Pairs[i].first == carreteras[j].init_node && od_Pairs[i].second == carreteras[j].term_node) { // ver que carretera j coincide con el odPAir[i] para asignar a dicha carretera j la demanda i  (el proceso de obtencionn de odPair y demandas de cada carretera tiene un orden distinto al proceso de la obtenciion de la lista de carreteras)
                            done = true;
                            strategyvec[j] = demands[i];
                        }
                    }
                }
            }
            Strategy_vector.push_back(strategyvec);
            /*if (camino == 0) {
                Strategy_vector.push_back(strategyvec);
            }
            else if (true || dot_product(strategyvec, Freeflowtimes) < multFactor * dot_product(Strategy_vector[0], Freeflowtimes)) { // para quedarse solo con las mejores estrategias para cada odPAir y evitar congestionar el strategy vectors
                Strategy_vector.push_back(strategyvec);
            }*/
        }
        Strategy_vectors.push_back(Strategy_vector);
    }
    return Strategy_vectors;
}
    

    /*
    if 
    mult_factor is None:
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
    for i in range(len(OD_pairs)): // 24*24 origen-destino trayectos
        Strategy_vectors[i] = list()
        OD_pair = np.array(OD_pairs[i])
        paths = k_shortest_paths(Networkx, str(OD_pair[0]), str(OD_pair[1]), K, weight = 'weight') // para cada odPAir se encuentran los k=5 caminos mas cortos
        for a in range(len(paths)):
            vec = np.zeros((E,1)) // un vector --> para cada carretera el valor 0
            for n in range(len(paths[a])-1):
                idx = get_edge_idx(Edges,  paths[a][n], paths[a][n+1])
                vec[idx] = 1 //identifica las carreteras usadas en el camino a (cada uno de los k caminos mas cortos)
            
            strategy_vec = np.multiply(vec, Demands[i]) //tamaño E = numCarreteras. Se calcula un strategy vector para cada uno de los 5 caminos mas cortos dado un odPAir
            if a == 0:
                Strategy_vectors[i].append(  strategy_vec )
            if a > 0 and np.dot(strategy_vec.T, Freeflowtimes) < multipl_factor* np.dot(Strategy_vectors[i][0].T, Freeflowtimes ):
                Strategy_vectors[i].append(strategy_vec )
                
       */

int dot_product(std::vector<int> vec1, std::vector<int> vec2) {
    int result = 0.0;
    for (int i = 0; i < vec1.size(); i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

std::vector<double> getCapacities(const NetworkData &n)
{
    std::vector<Carretera> carreteras = n.getCarreteras();
    std::vector<double> capacities (carreteras.size());
    for (int j = 0; j < carreteras.size(); j++)
        capacities[j] = carreteras[j].capacity;

    return capacities;
}

std::vector<int> getFreeFlowTimes(const NetworkData& n)
{
    std::vector<Carretera> carreteras = n.getCarreteras();
    std::vector<int> times(carreteras.size());
    for (int j = 0; j < carreteras.size(); j++)
        times[j] = carreteras[j].freeFlowTime;

    return times;
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



/*std::vector<double>*/ void Compute_traveltimes(const NetworkData & network, const std::vector< std::vector<std::vector<int>>>&Strategy_vectors, const std::vector<int>&played_actions, int player_id, std::vector<double> Capacities) {
    /*int N = Strategy_vectors.size(); // numero de jugadores
    std::vector<double> Total_occupancies(Strategy_vectors[0].size(), 0.0);
    for (int i = 0; i < N; ++i) { // Esto es un poco duda pero creo que esta bien
        std::vector<int> strategy_player = Strategy_vectors[i];
        int ind = played_actions[i];
        Total_occupancies[ind] += strategy_player[ind];
        
    }

    if (Capacities.empty()) {
        Capacities = getCapacities(network);
    }

    std::vector<Carretera> carreteras = network.getCarreteras();

    int E = carreteras.size();
    std::vector<double> a(E, 0.0);

    for (int i = 0; i < E; ++i) {
        a[i] = carreteras[i].freeFlowTime;
    }
    std::vector<double> b(E, 0.0);
    for (int i = 0; i < E; ++i) {
        b[i] = a[i] * 0.15 / std::pow(Capacities[i], carreteras[i].power);
    }
    std::vector<double> unit_times(E, 0.0);
    for (int i = 0; i < E; ++i) {
        unit_times[i] = a[i] + b[i] * std::pow(Total_occupancies[i], carreteras[i].power);
    }

    std::vector<double> Traveltimes(N, 0.0);
    if (player_id == -1) { // se hace para todos los jugadores
        for (int i = 0; i < N; ++i) {
            int a = played_actions[i];
            int asfd = Strategy_vectors[i][played_actions[i]];
            std::vector<double> X_i(E);
            // PROBLEMA AQUI: Comparar con Git + Entender porque funciona así
            for (int j = 0; j < E; j++)  
                X_i[j] = Strategy_vectors[played_actions[i]][j];
            // x_i sera la x de cada jugador, j es la carretera y se iguala a strategyvectors[accionjugadaporjugador(de 204 jugs)][demanda carretera de ese jugador]
            for (int j = 0; j < E; ++j) {
                Traveltimes[i] += X_i[j] * unit_times[j]; // se multiplica unit times x x: basicamente unit times es un factor para calcular el timepo de viaje
            }
        }
    }
    else {
        double X_i = Strategy_vectors[player_id][played_actions[player_id]];
        for (int j = 0; j < E; ++j) {
            Traveltimes[0] += X_i * unit_times[j];
        }
    }
    return Traveltimes;*/
}
