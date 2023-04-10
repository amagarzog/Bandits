/*#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace Eigen;

// Definir el modelo
struct Model {
    int num_points;
    VectorXd x_values, y_values;
    double sigma_e;

    // Función para calcular los residuos
    int operator()(const VectorXd& params, VectorXd& residuals) const {
        double alpha = params[0];
        double beta = params[1];
        double gamma = params[2];

        for (int i = 0; i < num_points; i++) {
            double x = x_values(i);
            double y = y_values(i);
            double y_fit = alpha * exp(-beta * x) + gamma;

            residuals(i) = (y - y_fit) / sigma_e;
        }

        return 0;
    }

    // Función para calcular los valores y errores de los parámetros
    void fit() {
        // Establecer los parámetros iniciales
        Vector3d params;
        params << 1.0, 1.0, 1.0;

        // Definir la estructura de datos para la optimización
        NumericalDiff<Model> numer_diff(*this);
        LevenbergMarquardt<NumericalDiff<Model>> lm(numer_diff);
        lm.parameters.maxfev = 1000;

        // Realizar la optimización
        lm.minimize(params);

        // Establecer los valores finales de los parámetros
        double alpha = params[0];
        double beta = params[1];
        double gamma = params[2];

        // Calcular el error en los parámetros
        VectorXd errors = lm.parameterErrors();

        double alpha_err = errors[0];
        double beta_err = errors[1];
        double gamma_err = errors[2];

        std::cout << "alpha = " << alpha << " +/- " << alpha_err << std::endl;
        std::cout << "beta = " << beta << " +/- " << beta_err << std::endl;
        std::cout << "gamma = " << gamma << " +/- " << gamma_err << std::endl;
    }
};

int main() {
    // Crear los datos de prueba
    int num_points = 10;
    VectorXd x_values = VectorXd::LinSpaced(num_points, 0.0, 1.0);
    VectorXd y_values = VectorXd::Random(num_points);
    double sigma_e = 0.1;

    // Crear el modelo y ajustarlo
    Model model;
    model.num_points = num_points;
    model.x_values = x_values;
    model.y_values = y_values;
    model.sigma_e = sigma_e;

    model.fit();

    return 0;
}
*/