// Librerie
#include "WaveSolver.hpp"
#include <cmath>
#include <random>
#include <iostream>

// Funzione per generare numeri casuali in virgola mobile
double random_double(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Generatore Mersenne Twister
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Funzione per generare numeri casuali interi
int random_int(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Generatore Mersenne Twister
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

int main() {
    // Generazione casuale dei parametri della simulazione
    double Lx = random_double(5.0, 7.0);   // Dimensione del dominio Lx
    double Ly = random_double(5.0, 7.0);   // Dimensione del dominio Ly
    unsigned int Nx = static_cast<unsigned int>(random_int(1, 15));  // Gradi di libertà Nx
    unsigned int Ny = static_cast<unsigned int>(random_int(1, 15));  // Gradi di libertà Ny
    double T = random_double(1.0, 3.0);    // Tempo totale T
    double dt = random_double(0.0001, 0.0003); // Passo temporale dt

    std::cout << "Domain size Lx=" << Lx << std::endl
              << "Domain size Ly=" << Ly << std::endl
              << "Degree of freedom Nx=" << Nx << std::endl
              << "Degree of freedom Ny=" << Ny << std::endl
              << "Total time=" << T << std::endl
              << "Time step dt=" << dt << std::endl;

    // Creazione del solver con parametri casuali
    WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);
    
    // Condizioni iniziali più robuste
    auto random_u0 = [](double x, double y) {
        if (x < 0 || x > 1 || y < 0 || y > 1) {
            return 0.0;  // Imposta a zero se i valori sono fuori intervallo
        }
        return std::sin(M_PI * x) * std::sin(M_PI * y);
    };

    auto random_u1 = [](double x, double y) {
        return 1e-6 * std::sin(M_PI * x) * std::sin(M_PI * y);  // Piccola velocità iniziale
    };

    // Impostazione delle condizioni iniziali
    solver.set_initial_conditions(random_u0, random_u1);

    // Funzione sorgente con maggiore controllo numerico
    auto random_source = [](double x, double y, double t) {
        if (std::abs(t) > 1000) {  // Limita t per evitare oscillazioni molto grandi
            return 0.0;
        }
        return std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(t);
    };

    // Impostazione della funzione sorgente
    solver.set_source_function(random_source);

    // Condizioni al contorno fisse
    auto random_boundary = [](double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
    solver.set_boundary_condition(random_boundary);

    // Risoluzione con Crank-Nicolson
    std::cout << "Solution with Crank-Nicolson...\n";
    solver.solve_crank_nicolson();
    //std::cout << "Completed\n";

    // Analisi delle prestazioni
    std::cout << "Performance analysis:\n";
    solver.analyze_performance();
    //std::cout << "Completed\n";
    std::cout << "Computed error: " << solver.compute_error() << std::endl;



    return 0;
}
