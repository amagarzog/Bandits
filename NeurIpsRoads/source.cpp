#include <iostream>
#include <vector>
#include <fstream>
#include <list>

//76 528 5 200

using namespace std;

std::vector<std::vector<int>> leerdatos() {
    int n;
    cin >> n;
    std::vector<std::vector<int>> mapa(n);
    int i = 0;
    for (int i = 0; i < n; i++) {
        int numConexiones;
        cin >> numConexiones;
        std::vector<int> conexiones(numConexiones);
        for (int j = 0; j < numConexiones; j++) {
            int nodo;
            cin >> nodo;
            conexiones[j] = nodo;
        }
        mapa[i] = conexiones;
    }
    return mapa;
}


void resuelveCaso(int i, int j) {
    std::vector<std::vector<int>> mapa;
    mapa = leerdatos();
    for (int j = 0; j < 3; j++) {
        cout << mapa[4][j] << "-";
    }

}



int main() {

    // ajuste para que cin extraiga directamente de un fichero
    std::ifstream in("sample.in");
    auto cinbuf = std::cin.rdbuf(in.rdbuf());

    resuelveCaso(1, 10);


    // restablecimiento de cin
    std::cin.rdbuf(cinbuf);
    //system("pause");
    return 0;
}