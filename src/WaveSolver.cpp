//libraries 
#include "WaveSolver.hpp"
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>      //for Jacobi preconditioner
#include <deal.II/lac/solver_gmres.h>      //for GMRES
#include <deal.II/lac/solver_bicgstab.h>   //for BiCGStab
#include <deal.II/base/logstream.h>        //for log management
#include <deal.II/numerics/data_out.h>     //for exporting numerical data
#include <deal.II/grid/grid_generator.h>   //for generating the grid
#include <deal.II/dofs/dof_tools.h>        //for handling degrees of freedom
#include <deal.II/lac/dynamic_sparsity_pattern.h>  //for managing dynamic sparsity pattern
#include <deal.II/numerics/vector_tools.h> //for vector operations
#include <deal.II/dofs/dof_handler.h>
#include <fstream>                         //for file output
#include <cmath>                           //for mathematical functions like sin
#include <chrono>                          //for measuring execution time
#include <iostream>                        //for console output
#include <iomanip>                         //for managing output precision
#include <omp.h>                           //for parallel execution
#include <map>                             //for managing maps (e.g., boundary values)
//end libraries 


//start initial function 
//class that represents the initial function, it is taken from the class Function from deal.II
class InitialFunction : public dealii::Function<2> {
public:
    //constructor that takes a function as a parameter 
    InitialFunction(const std::function<double(double, double)> &func) : func(func) {}

    //function that gives the value of the initial function in a point p 
    virtual double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
        return func(p[0], p[1]);  //calculates the value in base of the coordinates (x,y) 
    }

private:
    std::function<double(double, double)> func;  //function passed in the constructor
};
//end of initial function 


//start of constructor 

WaveSolver::WaveSolver(double Lx, double Ly, int Nx, int Ny, double T, double dt)
    : Lx(Lx), Ly(Ly), Nx(Nx), Ny(Ny), T(T), dt(dt), fe(1), dof_handler(triangulation)
{
    // Pulisci il triangulation (se necessario)
    triangulation.clear();

    // Crea una mesh suddivisa in Nx celle in x e Ny celle in y.
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  {static_cast<unsigned int>(Nx), static_cast<unsigned int>(Ny)},
                                                  dealii::Point<2>(0, 0),
                                                  dealii::Point<2>(Lx, Ly));


    // Stampa di debug: verifica il numero di celle attive
    std::cout << "Numero di celle attive: " << triangulation.n_active_cells() << std::endl;

    // Distribuisci i gradi di libertà sulla mesh appena creata
    dof_handler.distribute_dofs(fe);

    // (A questo punto, per elementi Q1, il numero di dof dovrebbe essere (Nx+1)*(Ny+1))
    std::cout << "Numero di dof: " << dof_handler.n_dofs() << std::endl;

    setup_system();
}

//end of constructor 

void WaveSolver::check_for_nan_in_vector(dealii::Vector<double>& vec) {
    for (unsigned int i = 0; i < vec.size(); ++i) {
        if (std::isnan(vec[i]) || std::isinf(vec[i])) {
            std::cerr << "NaN or Inf detected in vector at index " << i << std::endl;
            throw std::runtime_error("NaN or Inf detected in vector.");
        }
    }
}
void WaveSolver::check_for_empty_matrix(const dealii::SparseMatrix<double>& matrix) {
    // Verifica che la matrice non sia vuota usando i metodi m() e n()
    if (matrix.m() == 0 || matrix.n() == 0) {
        std::cerr << "Errore: la matrice di sistema è vuota!" << std::endl;
        std::exit(1);  // Uscita dal programma con codice di errore
    }

    // Aggiungi eventuali altre verifiche specifiche se necessario
}


double source_function(double x, double y, double t) {
    if (t == 0) {
        return std::sin(x + y);  // Se t = 0, non fare la divisione
    }
    return std::sin(x + y) / (t * t);
}






//start 
void WaveSolver::setup_system() {
    try {
        dof_handler.distribute_dofs(fe);

        dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        
        system_matrix.reinit(sparsity_pattern);
        rhs_vector.reinit(dof_handler.n_dofs());
        solution.reinit(dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        older_solution.reinit(dof_handler.n_dofs());
        
        // Inizializza la matrice di sistema e i vettori a valori validi
        solution = 0;
        old_solution = 0;
        older_solution = 0;
        rhs_vector = 0;
        
        // Assembla la matrice di sistema
        assemble_system_matrix();
        
        // Verifica che la matrice di sistema non sia vuota
        check_for_empty_matrix(system_matrix);
        
        // Verifica che non ci siano NaN nei vettori
        check_for_nan_in_vector(rhs_vector);
        check_for_nan_in_vector(solution);
        check_for_nan_in_vector(old_solution);
        check_for_nan_in_vector(older_solution);

    } catch (const std::exception& e) {
        std::cerr << "Error during setup_system: " << e.what() << std::endl;
        throw;
    }
}

void WaveSolver::assemble_system_matrix() {
    // Numero totale di gradi di libertà
    unsigned int n_dofs = dof_handler.n_dofs();

    // Costruisci il DynamicSparsityPattern in base al numero totale di dofs.
    dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);

    // Per ogni cella attiva, aggiungi le connessioni (non-zero) al pattern
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        std::vector<dealii::types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                dsp.add(local_dof_indices[i], local_dof_indices[j]);
    }

    // Copia il DynamicSparsityPattern in uno SparsityPattern (quello usato dalla matrice)
    sparsity_pattern.copy_from(dsp);

    // Ora inizializza la matrice sparsa globale con lo SparsityPattern ottenuto
    system_matrix.reinit(sparsity_pattern);

    // Costruzione della quadratura e FEValues
    dealii::QGauss<2> quadrature(3);
    dealii::FEValues<2> fe_values(fe, quadrature,
                                  dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Assemblaggio locale per ogni cella
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        if (!cell->is_locally_owned())
            continue;

        cell_matrix = 0;
        for (unsigned int q = 0; q < n_q_points; ++q) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                           fe_values.shape_grad(j, q) *
                                           fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                double value = cell_matrix(i, j);
                if (std::isnan(value) || std::isinf(value)) {
                    std::cerr << "NaN or Inf detected in cell_matrix[" << i << "," << j << "]!" << std::endl;
                    throw std::runtime_error("NaN or Inf detected in cell_matrix.");
                }
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], value);
            }
        }
    }
}



//start function set 
//function to set initial conditions for u0 (position) and u1 (velocity)
void WaveSolver::set_initial_conditions(const std::function<double(double, double)> &u0,
                                        const std::function<double(double, double)> &u1) {
    dealii::VectorTools::interpolate(dof_handler, InitialFunction(u0), solution);
    dealii::VectorTools::interpolate(dof_handler, InitialFunction(u1), old_solution);
    older_solution = old_solution;
}
//end 


//start boundary condition 
//function to set the boundary condition function
void WaveSolver::set_boundary_condition(const std::function<double(double, double, double)> &g) {
    boundary_condition = g;
}
//end 


//start source 
//function to set the source function
void WaveSolver::set_source_function(const std::function<double(double, double, double)> &f) {
    source_function = f;
}
//end 
void WaveSolver::apply_boundary_conditions(double time) {
    for (const auto &bc : boundary_values) {
        unsigned int boundary_index = bc.first;  // Indice del nodo di bordo
        double boundary_value = bc.second;       // Valore della condizione al contorno
        
        // Imposta la soluzione nei nodi di bordo
        solution(boundary_index) = boundary_value;

        // Pulisce la riga corrispondente nella matrice di sistema
        for (unsigned int j = 0; j < system_matrix.m(); ++j) {
            system_matrix(boundary_index, j) = 0.0;
        }

        // Imposta un 1 sulla diagonale per fissare il valore della soluzione
        system_matrix(boundary_index, boundary_index) = 1.0;

        // Aggiorna il termine noto
        rhs_vector(boundary_index) = boundary_value;
    }
}




void print_matrix(const dealii::SparseMatrix<double> &system_matrix) {
    std::cout << "System Matrix " << system_matrix.m() << "x" << system_matrix.n() << ":" << std::endl;
    /*for (unsigned int i = 0; i < system_matrix.m(); ++i) {
        for (unsigned int j = 0; j < system_matrix.n(); ++j) {
            std::cout << system_matrix.el(i, j) << "\t";
        }
        std::cout << "\n";
    }*/
}


void print_vector(const dealii::Vector<double> &rhs_vector) {
    std::cout << "\nRHS Vector:\n";
    for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
        std::cout << rhs_vector[i] << "\n";
    }
}


bool WaveSolver::solve_linear_system(dealii::SparseMatrix<double>& matrix, dealii::Vector<double>& sol, const dealii::Vector<double>& rhs) {

    dealii::SolverControl control(20000, 1e-6);
    dealii::SolverBicgstab<> solver(control);
    try {
        
        dealii::PreconditionJacobi<> preconditioner;
        preconditioner.initialize(matrix);
        
        solver.solve(matrix, sol, rhs, preconditioner);
        return true;
    }
    catch (std::exception &e) {
        std::cerr << "solve failed...\n";
        std::cout << "Iteration: " << control.last_step() 
                  << " Residual: " << control.last_value() << std::endl;
        return false;
    }
}


//start Crank-Nicolson method to solve the wave equation
void WaveSolver::solve_crank_nicolson() {
    try {
        dealii::Vector<double> laplacian(solution.size());
        dealii::Vector<double> rhs_vector_new(solution.size());
        dealii::Vector<double> rhs_vector_old(solution.size());
        
        // Creazione del file PVD per la raccolta dei file VTU
        std::ofstream pvd_file("solution.pvd");
        pvd_file << "<?xml version=\"1.0\"?>\n";
        pvd_file << "<VTUFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd_file << "<Collection>\n";

        // Stampa della matrice system_matrix per il debug
        print_matrix(system_matrix);

        // Loop per i passi temporali
        for (double t = 0; t < T; t += dt) {
            // Applicazione delle condizioni al contorno
            apply_boundary_conditions(t);

            // Calcolo del laplaciano della soluzione corrente (a metà passo, come per Crank-Nicolson)
            system_matrix.vmult(laplacian, solution);

            // Inizializzazione del nuovo vettore rhs (è la somma delle soluzioni precedenti)
            rhs_vector_new = laplacian;

            // Applicazione del termine sorgente, se presente
            if (source_function) {
                for (unsigned int i = 0; i < rhs_vector_new.size(); ++i) {
                    double x = dof_handler.get_fe().get_unit_support_points()[i][0];
                    double y = dof_handler.get_fe().get_unit_support_points()[i][1];
                    double source_val = source_function(x * Lx, y * Ly, t);
                    if (std::isnan(source_val)) {
                        std::cerr << "NaN detected in source function at time " << t << std::endl;
                        exit(1); // Uscita in caso di NaN nel termine sorgente
                    }
                    rhs_vector_new[i] += source_val * dt * dt / 2.0;  // Come Crank-Nicolson, usiamo una media
                }
            }

            // Calcoliamo il termine rhs per la soluzione precedente (vecchia soluzione)
            rhs_vector_old = rhs_vector_new; // Vettore rhs aggiornato per la soluzione precedente

            // Risoluzione del sistema lineare usando Bicgstab con precondizionatore Jacobi
            if (!solve_linear_system(system_matrix, solution, rhs_vector_old)) {
                throw std::runtime_error("Error during Crank-Nicolson: failed to solve linear system at time " + std::to_string(t));
            }

            // Aggiornamento delle soluzioni per il passo successivo
            older_solution.swap(old_solution);
            old_solution.swap(solution);

            // Scrittura del file VTU per il passo corrente
            std::string vtu_filename = "solution_" + std::to_string(int(t / dt)) + ".vtu";
            std::ofstream vtu_file(vtu_filename);
            dealii::DataOut<2> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();
            data_out.write_vtu(vtu_file);
            vtu_file.close();

            // Aggiunta del riferimento al file VTU nel file PVD
            pvd_file << "<DataSet timestep=\"" << t << "\" group=\"\" part=\"0\" file=\"" << vtu_filename << "\"/>\n";
        }

        // Chiusura del file PVD
        pvd_file << "</Collection>\n";
        pvd_file << "</VTUFile>\n";
        pvd_file.close();

    } catch (const std::exception &e) {
        std::cerr << "Error during Crank-Nicolson: " << e.what() << std::endl;
        exit(1);
    }
}

//end 



//start analyze_performance
void WaveSolver::analyze_performance() const {
    //benchmark function to measure the execution time of Crank-Nicolson method
    auto benchmark = [&](void (WaveSolver::*solve_method)(), const std::string &name) {
        // Measure the time taken by the solve method for a single run
        auto start = std::chrono::high_resolution_clock::now();
        (const_cast<WaveSolver *>(this)->*solve_method)(); // Call the solver method
        auto end = std::chrono::high_resolution_clock::now();
        
        // Output the time taken for the solver method
        double elapsed_time = std::chrono::duration<double>(end - start).count();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << name << " Scheme: " << elapsed_time << " sec\n";
    };

    // Benchmark the Crank-Nicolson method (one execution)
    benchmark(&WaveSolver::solve_crank_nicolson, "Crank-Nicolson");
}





double WaveSolver::compute_error() const {
    MappingQ1<2> mapping;
    QGauss<2> quadrature(3);
    Vector<double> error_per_cell(triangulation.n_active_cells());

    // Create an instance of the ExactSolution class
    ExactSolution exact_solution;
    
    // Compute the L2 norm difference between numerical and exact solutions
    VectorTools::integrate_difference(mapping, dof_handler, solution,
                                      exact_solution, error_per_cell,
                                      quadrature, VectorTools::L2_norm);

    return VectorTools::compute_global_error(triangulation, error_per_cell, VectorTools::L2_norm);
}


