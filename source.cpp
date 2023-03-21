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
    //std::vector<std::vector<std::string>> data = read_csv("SiouxFalls_node.csv");
    //print(data);
    //Initilize_Players_ini(); 

    return 0;
}