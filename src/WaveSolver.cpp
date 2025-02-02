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
//constructor of WaveSolver that sets up geometry, triangulation, and degree of freedom distribution
WaveSolver::WaveSolver(double Lx, double Ly, int Nx, int Ny, double T, double dt)
    : Lx(Lx), Ly(Ly), Nx(Nx), Ny(Ny), T(T), dt(dt), fe(1), dof_handler(triangulation) {
    dealii::GridGenerator::hyper_rectangle(triangulation, {0, 0}, {Lx, Ly});
    triangulation.refine_global(static_cast<unsigned int>(std::log2(Nx)));  //refine the grid
    setup_system();  //set up the system (matrices, vectors, etc.)
}
//end of constructor 


//start 
//function to prepare the linear system (dof_handler, matrix, vectors)
void WaveSolver::setup_system() {
    try {
        dof_handler.distribute_dofs(fe);
        dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        
        system_matrix.reinit(sparsity_pattern);
        //mass_matrix.reinit(sparsity_pattern);
        
        solution.reinit(dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        older_solution.reinit(dof_handler.n_dofs());
        rhs_vector.reinit(dof_handler.n_dofs());
        
        assemble_system_matrix();
    } catch (const std::exception& e) {
        std::cerr << "Error in setup_system: " << e.what() << std::endl;
        throw;
    }
}
//end 

//Assembles the system matrix
void WaveSolver::assemble_system_matrix() {
    system_matrix = 0;
    
    dealii::QGauss<2> quadrature(3);
    dealii::FEValues<2> fe_values(fe, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();
    
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        
        for (unsigned int q = 0; q < n_q_points; ++q) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *fe_values.shape_grad(j, q) *fe_values.JxW(q);
                }
            }
        }
        
        cell->get_dof_indices(local_dof_indices);
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
        }
    }
}
//end


//start reset function 
//function to reset the solution vectors (current and previous ones)
void WaveSolver::reset_solutions() {
    solution = 0;
    old_solution = 0;
    older_solution = 0;
    rhs_vector = 0;
}
//end 


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


//start apply 
//function made to apply boundary conditions to the system 
void WaveSolver::apply_boundary_conditions(double time) {
    if (boundary_condition) {
        std::map<dealii::types::global_dof_index, double> boundary_values;

        // Use a Function object for boundary values if possible.  It's more efficient.
        class BoundaryFunction : public dealii::Function<2> {
        public:
            BoundaryFunction(const std::function<double(double, double, double)>& bc, double time) : bc(bc), time(time) {}
            virtual double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
                return bc(p[0], p[1], time);
            }
        private:
            const std::function<double(double, double, double)>& bc;
            double time;
        };
        dealii::VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryFunction(boundary_condition, time), boundary_values);
        for (const auto &entry : boundary_values) {
            solution[entry.first] = entry.second;
            rhs_vector[entry.first] = entry.second;  // Is this really necessary?  Often, boundary conditions are handled *after* the system assembly.
        }
    }
}
//end 


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


//start Newmark method to solve the wave equation
void WaveSolver::solve_newmark() {
    try {
        dealii::Vector<double> laplacian(solution.size());

        // Creazione del file PVD per la raccolta dei file VTU
        std::ofstream pvd_file("solution.pvd");
        pvd_file << "<?xml version=\"1.0\"?>\n";
        pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd_file << "<Collection>\n";

        print_matrix(system_matrix);

        for (double t = 0; t < T; t += dt) {
            apply_boundary_conditions(t);

            // Calcolo del laplaciano della soluzione corrente
            system_matrix.vmult(laplacian, solution);
            rhs_vector=laplacian;

            // Applicazione del termine sorgente, se presente
            if (source_function) {
                for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                    double x = dof_handler.get_fe().get_unit_support_points()[i][0];
                    double y = dof_handler.get_fe().get_unit_support_points()[i][1];
                    rhs_vector[i] += source_function(x*Lx, y*Ly, t) * dt * dt;
                }
            }

            // Risoluzione del sistema lineare usando Bicgstab con precondizionatore Jacobi
            if (!solve_linear_system(system_matrix, solution, rhs_vector)) {
                throw std::runtime_error("Error during Newmark: failed to solve linear system at time " +  std::to_string(t));
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
        pvd_file << "</VTKFile>\n";
        pvd_file.close();
    } catch (const std::exception &e) {
        std::cerr << "Error during Newmark: " << e.what() << std::endl;
        exit(1);
    }
}
//end 


//start Crank-Nicolson method to solve the wave equation
void WaveSolver::solve_crank_nicolson() {
    try {
        dealii::Vector<double> laplacian(solution.size());

        // Creazione del file PVD per ParaView
        std::ofstream pvd_file("solution.pvd");
        pvd_file << "<?xml version=\"1.0\"?>\n";
        pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd_file << "<Collection>\n";

        // Precompute SSOR preconditioner only once
        dealii::PreconditionJacobi<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        for (double t = 0; t < T; t += dt) {
            apply_boundary_conditions(t);

            system_matrix.vmult(laplacian, solution);

            for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                rhs_vector[i] += laplacian[i] * dt * dt / 2.0;
            }

            if (source_function) {
                for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                    double x = dof_handler.get_fe().get_unit_support_points()[i][0];
                    double y = dof_handler.get_fe().get_unit_support_points()[i][1];
                    rhs_vector[i] += source_function(x*Lx, y*Ly, t) * dt * dt / 2.0;
                }
            }

            // Risoluzione del sistema lineare usando Bicgstab con precondizionatore Jacobi
            if (!solve_linear_system(system_matrix, solution, rhs_vector)) {
                throw std::runtime_error("Error during Newmark: failed to solve linear system at time " +  std::to_string(t));
            }

            older_solution.swap(old_solution);
            old_solution.swap(solution);

            // Nome del file VTU per la soluzione a questo tempo
            std::string vtu_filename = "solution_" + std::to_string(int(t / dt)) + ".vtu";

            // Scrittura della soluzione nel file VTU
            std::ofstream vtu_file(vtu_filename);
            dealii::DataOut<2> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();
            data_out.write_vtu(vtu_file);

            // Aggiunta al file PVD
            pvd_file << "<DataSet timestep=\"" << t << "\" group=\"\" part=\"0\" file=\"" 
                     << vtu_filename << "\"/>\n";
        }

        // Chiusura del file PVD
        pvd_file << "</Collection>\n";
        pvd_file << "</VTKFile>\n";
        pvd_file.close();

    } catch (const std::exception &e) {
        std::cerr << "Error during Crank-Nicolson: " << e.what() << std::endl;
        exit(1);
    }
};
//end 
//end 


//function to produce output in vtk file 
void WaveSolver::output_vtk(double time) {
    constexpr unsigned int dim = 2;
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();  // Build patches only once
    std::string vtk_filename = "solution-" + std::to_string(time) + ".vtk";
    std::ofstream vtk_file(vtk_filename);
    data_out.write_vtk(vtk_file);
}
//end


//start calculate the error (L2 and L∞ norms)
double WaveSolver::calculate_error() {
    double error_l2 = 0.0, error_linf = 0.0;
    double dx = Lx / Nx, dy = Ly / Ny;

    for (unsigned int i = 0; i < Nx; ++i) {
        for (unsigned int j = 0; j < Ny; ++j) {
            double x = i * dx, y = j * dy;
            //compute the exact solution
            double exact_value = std::sin(M_PI * x) * std::sin(M_PI * y);
            double diff = solution(i + j * Nx) - exact_value;
            
            //update the L2 and L∞ errors
            error_l2 += diff * diff * dx * dy;
            error_linf = std::max(error_linf, std::abs(diff));
        }
    }

    //compute the L2 norm
    error_l2 = std::sqrt(error_l2);
    
    //handle errors (if any) such as infinite or NaN values
    if (std::isinf(error_l2) || std::isinf(error_linf)) {
        std::cout << "Error: Inf\n";
    } else if (std::isnan(error_l2) || std::isnan(error_linf)) {
        std::cout << "Error: NaN\n";
    } else {
        //print the L2 and L∞ errors with fixed precision
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "L2 Error: " << error_l2 << ", L∞ Error: " << error_linf << std::endl;
    }

    return error_l2;
}
//end 



//start test convergence 
void WaveSolver::test_convergence() {
    int Nx_start = 50, Ny_start = 50;  //store initial values
    double Lx = 10.0, Ly = 10.0, T = 1.0, dt_start = 0.01; //start with a smaller dt

    for (int i = 0; i < 5; ++i) {
        int Nx = Nx_start * std::pow(2, i);  //calculate Nx and Ny for this level
        int Ny = Ny_start * std::pow(2, i);
        double dt = dt_start / std::pow(2, i); //reduce dt as grid refines crucial for stability

        WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);  //recreate solver each time

        solver.set_initial_conditions(
            [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); },
            [](double x, double y) { return 0.0; }
        );

        solver.solve_newmark(); //or solve_crank_nicolson()

        double error = solver.calculate_error();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Error for Nx = " << Nx << ", Ny = " << Ny << ", dt = " << dt << ": " << error << std::endl;
    }
}
//end


//start 
void WaveSolver::analyze_performance() const {
    const int num_runs = 5;
    //benchmark function to measure the execution time of different solving methods (performance)
    auto benchmark = [&](void (WaveSolver::*solve_method)(), const std::string &name) {
        double total_time = 0.0;
        for (int i = 0; i < num_runs; ++i) {
            //measure the time taken by the solve method
            auto start = std::chrono::high_resolution_clock::now();
            (const_cast<WaveSolver *>(this)->*solve_method)(); //correct here
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double>(end - start).count();
        }
        //output the average time taken for the solver method
        std::cout << std::fixed << std::setprecision(6);
        std::cout << name << " Scheme: " << (total_time / num_runs) << " sec\n";
    };

    //benchmark the Newmark and Crank-Nicolson methods
    benchmark(&WaveSolver::solve_newmark, "Newmark");
    benchmark(&WaveSolver::solve_crank_nicolson, "Crank-Nicolson");
}
//end 
