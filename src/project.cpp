#include "WaveSolver.hpp"
#include <cmath>
#include <random>

// Funzione per generare un numero casuale in un intervallo [min, max]
double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}
//Fine funzione 



//Inizio main
//Chiamo le funzioni/metodi creti nel file header e nel file cpp 
int main() {
     // Generazione casuale di parametri principali
    const double Lx = random_double(5.0, 20.0);  // Dimensione dominio Lx
    const double Ly = random_double(5.0, 20.0);  // Dimensione dominio Ly
    const int Nx = static_cast<int>(random_double(10, 100));  // Gradi di libertà Nx
    const int Ny = static_cast<int>(random_double(10, 100));  // Gradi di libertà Ny
    const double T = random_double(0.5, 5.0);  // Tempo totale T
    const double dt = random_double(0.01, 0.2);  // Passo temporale dt

    // Creazione del solver con parametri casuali
    WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);

    // Generazione casuale di condizioni iniziali u0 e u1
    solver.set_initial_conditions(
        [](double x, double y) { 
            return std::sin(random_double(0.1, 2.0) * M_PI * x) * std::sin(random_double(0.1, 2.0) * M_PI * y); 
        },
        [](double x, double y) { 
            return random_double(0.0, 1.0) * std::cos(random_double(0.1, 2.0) * M_PI * x); 
        }
    );

    // Generazione casuale della funzione sorgente
    auto source_function = [](double x, double y, double t) {
        return std::sin(random_double(0.1, 2.0) * M_PI * x) * 
               std::sin(random_double(0.1, 2.0) * M_PI * y) * 
               std::cos(random_double(0.1, 2.0) * M_PI * t);
    };
    solver.set_source_function(source_function);

    // Generazione casuale delle condizioni al contorno
    auto boundary_condition = [](double x, double y, double t) {
        return random_double(0.0, 1.0) * std::sin(random_double(0.1, 2.0) * M_PI * t);
    };

    solver.set_boundary_condition(boundary_condition);

    // Risolvi con il metodo Newmark
    solver.solve_newmark();

    // Calcola l'errore
    solver.calculate_error();

    // Test di convergenza
    solver.test_convergence();

    solver.analyze_performance();

    return 0;
}
