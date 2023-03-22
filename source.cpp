#include "network.h"

void print(std::vector<std::vector<std::string>> d) {
    for (auto fila : d) {
        for (auto campo : fila) {
            std::cout << campo << " ";
        }
        std::cout << std::endl;
    }
}



int main() {
    NetworkData network = createNetwork();
    std::vector<std::vector<OD_Demand>> od_Demands = createOD_Demands(); //Es una matriz cuadrada que relaciona las demandas de un nodo i (fila) a un nodo j (columna)
    return 0;
}