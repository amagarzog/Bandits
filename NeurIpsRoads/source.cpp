#include "network.h"


int main() {
    std::vector<std::vector<int>> OD_Demands = takeDemands();
    NetworkData network = createNetwork();
    std::vector<std::vector<int>> strategyV = computeStrategyVectors(OD_Demands, network);




    return 0;
}