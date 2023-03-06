#include <iostream>
#include <vector>
#include <fstream>
#include <list>

//76 528 5 200

using namespace std;

vector<vector<int>> leerdatos(){
    int n;
    cin >> n;
    vector<vector<int>> mapa(n);
    int i = 0;
    for(int i = 0; i < n; i++){
        int numConexiones;
        cin >> numConexiones;
        vector<int> conexiones(numConexiones);
        for(int j = 0; j < numConexiones; j++){
            int nodo;
            cin >> nodo; 
            conexiones[j] = nodo;
        }
        mapa[i] = conexiones;
    }
}


void resuelveCaso(int i, int j) {
    vector<vector<int>> mapa = leerdatos();
    cout << "holamundo" << endl;
    for(int j = 0; j < 3; j++){
            cout << mapa[4][0];
            cout << "holamundo" << endl;
    }

}



int main() {

    // ajuste para que cin extraiga directamente de un fichero
    std::ifstream in("sample.in");
    auto cinbuf = std::cin.rdbuf(in.rdbuf());

    cout << "holamundo" << endl;
    resuelveCaso(1, 10);


    // restablecimiento de cin
    std::cin.rdbuf(cinbuf);
    //system("pause");
    return 0;
}