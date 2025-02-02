//libraries
#include "WaveSolver.hpp"
#include <cmath>
#include <random>
//end 


//random number generation utility
double random_double(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd()); //Mersenne Twister generator
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}
//end


//start
int random_int(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd()); //Mersenne Twister generator
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}
//end 


//start main 
int main() {
   //randomly generating the simulation parameters
    double Lx = random_double(5.0, 7.0);   //domain size Lx
    double Ly = random_double(5.0, 7.0);   //domain size Ly
    int Nx = random_int(1, 15);           //degrees of freedom Nx
    int Ny = random_int(1, 15);           //degrees of freedom Ny
    double T = random_double(1, 3.0);      //total time T
    double dt = random_double(0.0001, 0.0003);   //time step dt

    std::cout << "domain size Lx=" << Lx << std::endl
              << "domain size Ly=" << Ly << std::endl
              << "degree of freedom Nx=" << Nx << std::endl
              << "degree of freedom Ny=" << Ny << std::endl
              << "total time=" << T << std::endl
              << "time step dt=" << dt << std::endl;

    //creation of the solver with random parameters 
    WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);
    

    // Condizioni iniziali più robuste
auto random_u0 = [](double x, double y) {
    // Assicurati che x e y siano limitati
    if (x < 0 || x > 1 || y < 0 || y > 1) {
        return 0.0;  // Imposta a zero se i valori sono fuori dall'intervallo
    }
    return std::sin(M_PI * x) * std::sin(M_PI * y);
};

auto random_u1 = [](double x, double y) {
    return 1e-6 * std::sin(M_PI * x) * std::sin(M_PI * y);  // Velocità iniziale piccola
};


// Impostazione delle condizioni iniziali
solver.set_initial_conditions(random_u0, random_u1);
// Funzione sorgente con maggiore controllo numerico
auto random_source = [](double x, double y, double t) {
    if (std::abs(t) > 1000) {  // Limita il valore di t per evitare oscillazioni molto grandi
        return 0.0;
    }
    return std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(t);
};

// Impostazione della funzione sorgente
solver.set_source_function(random_source);

    //boundary conditions
    auto random_boundary = [](double x, double y, double t) {
        return 0.0;  //fixed 
    };
    solver.set_boundary_condition(random_boundary);

    std::cout << "Solution with Newmark...\n";
    solver.solve_newmark();
    std::cout << "Completed\n";

    std::cout << "Solution with Crank-Nicolson...\n";
    solver.solve_crank_nicolson();
    std::cout << "Completed\n";

    //analysis of perfomance 
    std::cout << "Analisi delle prestazioni:\n";
    solver.analyze_performance();
    std::cout << "Completed\n";

    //convergence test 
    std::cout << "Convergence test:\n";
    solver.test_convergence();
    std::cout << "Completed\n";

    return 0;
}
//end main 