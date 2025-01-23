#include "WaveSolver.hpp"
#include <deal.II/base/logstream.h>        // Per la gestione dei log
#include <deal.II/numerics/data_out.h>     // Per esportare i dati numerici
#include <deal.II/grid/grid_generator.h>   // Per generare la griglia
#include <deal.II/dofs/dof_tools.h>        // Per gestire i gradi di libertà
#include <deal.II/lac/dynamic_sparsity_pattern.h>  // Per la gestione della struttura sparsa dinamica
#include <deal.II/numerics/vector_tools.h> // Per le operazioni sui vettori
#include <fstream>                         // Per l'output su file
#include <cmath>                           // Per funzioni matematiche come sin
#include <chrono>                          // Per la misurazione del tempo di esecuzione
#include <iostream>                        // Per l'output su console
#include <iomanip>                         // Per gestire la precisione nell'output
#include <omp.h>                           // Per l'esecuzione parallela
#include <map>                             // Per la gestione di mappe (es. per i valori al contorno)

// Classe che rappresenta la funzione iniziale, derivata dalla classe Function di deal.II
class InitialFunction : public dealii::Function<2> {
public:
    // Costruttore che prende una funzione come parametro
    InitialFunction(const std::function<double(double, double)> &func) : func(func) {}

    // Funzione che restituisce il valore della funzione iniziale in un punto p
    virtual double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
        return func(p[0], p[1]);  // Calcola il valore in base alle coordinate (x, y)
    }

private:
    std::function<double(double, double)> func;  // Funzione passata al costruttore
};

// Costruttore di WaveSolver che imposta la geometria, la triangolazione e la distribuzione dei gradi di libertà
WaveSolver::WaveSolver(double Lx, double Ly, int Nx, int Ny, double T, double dt)
    : Lx(Lx), Ly(Ly), Nx(Nx), Ny(Ny), T(T), dt(dt), fe(1), dof_handler(triangulation) {
    // Genera una griglia rettangolare nel dominio definito da (0,0) a (Lx, Ly)
    dealii::GridGenerator::hyper_rectangle(triangulation, {0, 0}, {Lx, Ly});
    triangulation.refine_global(static_cast<unsigned int>(std::log2(Nx)));  // Rifinisce la griglia
    setup_system();  // Imposta il sistema (matrici, vettori, ecc.)
}

// Funzione per preparare il sistema lineare (dof_handler, matrice, vettori)
void WaveSolver::setup_system() {
    // Distribuisce i gradi di libertà tra i nodi
    dof_handler.distribute_dofs(fe);

    // Crea una struttura sparsa dinamica per la matrice del sistema
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);  // Crea il pattern di sparseness
    sparsity_pattern.copy_from(dsp);  // Copia il pattern nella sparsity_pattern
    system_matrix.reinit(sparsity_pattern);  // Inizializza la matrice sparsa

    // Inizializza i vettori delle soluzioni (vecchie, nuove, e rhs)
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    older_solution.reinit(dof_handler.n_dofs());
    rhs_vector.reinit(dof_handler.n_dofs());

    reset_solutions();  // Resetta le soluzioni ai valori iniziali (tutti a zero)
}

// Funzione per resettare i vettori delle soluzioni (vecchie e attuali)
void WaveSolver::reset_solutions() {
    solution = 0;
    old_solution = 0;
    older_solution = 0;
    rhs_vector = 0;
}

// Impostazione delle condizioni iniziali per u0 (posizione) e u1 (velocità)
void WaveSolver::set_initial_conditions(const std::function<double(double, double)> &u0,
                                        const std::function<double(double, double)> &u1) {
    // Interpola la funzione u0 nei gradi di libertà per ottenere la posizione iniziale
    dealii::VectorTools::interpolate(dof_handler, InitialFunction(u0), solution);
    // Interpola la funzione u1 nei gradi di libertà per ottenere la velocità iniziale
    dealii::VectorTools::interpolate(dof_handler, InitialFunction(u1), old_solution);
    older_solution = old_solution;  // La soluzione precedente è quella attuale
}

// Impostazione della funzione per le condizioni al contorno
void WaveSolver::set_boundary_condition(const std::function<double(double, double, double)> &g) {
    boundary_condition = g;
}

// Impostazione della funzione sorgente
void WaveSolver::set_source_function(const std::function<double(double, double, double)> &f) {
    source_function = f;
}

// Funzione per applicare le condizioni al contorno al sistema
void WaveSolver::apply_boundary_conditions(double time) {
    if (boundary_condition) {
        // Mappa per i valori al contorno
        std::map<dealii::types::global_dof_index, double> boundary_values;
        dealii::VectorTools::interpolate_boundary_values(
            dof_handler, 0, dealii::Functions::ConstantFunction<2>(0.0), boundary_values);

        // Applica i valori al contorno a tutti i gradi di libertà
        for (const auto &entry : boundary_values) {
            const dealii::types::global_dof_index dof_index = entry.first;
            const double boundary_value = entry.second;

            solution[dof_index] = boundary_value;
            rhs_vector[dof_index] = boundary_value;
        }
    } 
}

// Metodo per risolvere l'equazione usando il metodo di Newmark
void WaveSolver::solve_newmark() {
    // Ciclo temporale
    for (double t = 0; t < T; t += dt) {
        apply_boundary_conditions(t);  // Applica le condizioni al contorno al passo temporale corrente

        dealii::Vector<double> laplacian = solution;
#pragma omp parallel
        {
            // Calcola il prodotto matrice-vettore
            system_matrix.vmult(laplacian, solution);
        }

        // Aggiorna il vettore rhs con il termine della laplaciana moltiplicato per dt^2
        for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
            rhs_vector[i] += laplacian[i] * dt * dt;
        }

        // Se esiste una funzione sorgente, la aggiunge al vettore rhs
        if (source_function) {
            for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                rhs_vector[i] += source_function(dof_handler.get_fe().get_unit_support_points()[i][0], 
                                                  dof_handler.get_fe().get_unit_support_points()[i][1], t) * dt * dt;
            }
        } 

        // Risoluzione del sistema lineare con il metodo Conjugate Gradient
        dealii::SolverControl solver_control(1000, 1e-12);  // Controllo del solver
        dealii::SolverCG<> solver(solver_control);  // Solver Conjugate Gradient
        dealii::PreconditionSSOR<> preconditioner;   // Precondizionatore SSOR
        preconditioner.initialize(system_matrix, 1.2);  // Inizializzazione del precondizionatore
        solver.solve(system_matrix, solution, rhs_vector, preconditioner);  // Risoluzione

        // Aggiorna le soluzioni precedenti
        older_solution = old_solution;
        old_solution = solution;
    }
}

// Metodo per risolvere l'equazione usando il metodo di Crank-Nicolson
void WaveSolver::solve_crank_nicolson() {
    // Ciclo temporale
    for (double t = 0; t < T; t += dt) {
        apply_boundary_conditions(t);  // Applica le condizioni al contorno

        dealii::Vector<double> laplacian = solution;
#pragma omp parallel
        {
            // Calcola il prodotto matrice-vettore
            system_matrix.vmult(laplacian, solution);
        }

        // Aggiorna il vettore rhs con il termine della laplaciana
        for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
            rhs_vector[i] += laplacian[i] * dt * dt / 2.0;
        }

        // Aggiunge il termine sorgente al rhs
        for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
            rhs_vector[i] += source_function(dof_handler.get_fe().get_unit_support_points()[i][0], 
                                              dof_handler.get_fe().get_unit_support_points()[i][1], t) * dt * dt / 2.0;
        }

        // Risoluzione del sistema con Conjugate Gradient
        dealii::SolverControl solver_control(1000, 1e-12);
        dealii::SolverCG<> solver(solver_control);
        dealii::PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);
        solver.solve(system_matrix, solution, rhs_vector, preconditioner);

        // Aggiorna le soluzioni precedenti
        older_solution = old_solution;
        old_solution = solution;
    }
}

// Calcola l'errore L2 e L∞ rispetto a una soluzione esatta (qui usata come esempio sin(M_PI * x) * sin(M_PI * y))
double WaveSolver::calculate_error() {
    double error_l2 = 0.0;
    double error_linf = 0.0;

    // Ciclo su tutti i gradi di libertà per calcolare l'errore
    for (unsigned int i = 0; i < solution.size(); ++i) {
        double exact_value = std::sin(M_PI * solution(i));  // Funzione esatta come esempio
        double diff = solution(i) - exact_value;
        error_l2 += diff * diff;
        error_linf = std::max(error_linf, std::abs(diff));
    }

    error_l2 = std::sqrt(error_l2);  // Errore L2
    std::cout << "Errore L2: " << error_l2 << ", Errore L∞: " << error_linf << std::endl;

    return error_l2;
}

// Funzione per testare la convergenza variando la risoluzione della griglia
void WaveSolver::test_convergence() {
    int Nx = 50, Ny = 50;
    double Lx = 10.0, Ly = 10.0, T = 1.0, dt = 0.1;

    for (int i = 0; i < 5; ++i) {
        WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);

        solver.set_initial_conditions([](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); },
                                      [](double x, double y) { return 0.0; });

        solver.solve_newmark();  // Risolvi con Newmark
        double error = solver.calculate_error();  // Calcola errore L2 e L∞
        std::cout << "Errore per Nx = " << Nx << ", Ny = " << Ny << ": " << error << std::endl;

        Nx *= 2;
        Ny *= 2;
    }
}

// Funzione per analizzare le performance di entrambi i metodi
void WaveSolver::analyze_performance() const {
    const int num_runs = 10;

    double total_time_newmark = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const_cast<WaveSolver *>(this)->solve_newmark();
        auto end = std::chrono::high_resolution_clock::now();
        total_time_newmark += std::chrono::duration<double>(end - start).count();
    }

    double total_time_crank_nicolson = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const_cast<WaveSolver *>(this)->solve_crank_nicolson();
        auto end = std::chrono::high_resolution_clock::now();
        total_time_crank_nicolson += std::chrono::duration<double>(end - start).count();
    }

    // Mostra il tempo medio per ciascun metodo
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Performance Analysis:\n";
    std::cout << "Newmark Scheme          : " << (total_time_newmark / num_runs) << " seconds\n";
    std::cout << "Crank-Nicolson Scheme   : " << (total_time_crank_nicolson / num_runs) << " seconds\n";
}
