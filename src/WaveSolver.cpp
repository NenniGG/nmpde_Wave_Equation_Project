//libraries 
#include "WaveSolver.hpp"
#include <deal.II/lac/vector.h>
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
    dof_handler.distribute_dofs(fe);
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    older_solution.reinit(dof_handler.n_dofs());
    rhs_vector.reinit(dof_handler.n_dofs());
    reset_solutions();
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
        //map for boundary values 
        std::map<dealii::types::global_dof_index, double> boundary_values;
        dealii::VectorTools::interpolate_boundary_values(
            dof_handler, 0, dealii::Functions::ConstantFunction<2>(0.0), boundary_values);

        //apply boundary values to every gol
        for (const auto &entry : boundary_values) {
            const dealii::types::global_dof_index dof_index = entry.first;
            const double boundary_value = entry.second;

            solution[dof_index] = boundary_value;
            rhs_vector[dof_index] = boundary_value;
        }
    } 
}
//end 


//start Newmark method to solve the wave equation
void WaveSolver::solve_newmark() {
    try {
        dealii::Vector<double> laplacian(solution.size());
        
        for (double t = 0; t < T; t += dt) {
            apply_boundary_conditions(t);

            #pragma omp parallel
            {
                system_matrix.vmult(laplacian, solution);
            }

            #pragma omp parallel for
            for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                rhs_vector[i] += laplacian[i] * dt * dt;
            }

            if (source_function) {
                #pragma omp parallel for
                for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                    double x = dof_handler.get_fe().get_unit_support_points()[i][0];
                    double y = dof_handler.get_fe().get_unit_support_points()[i][1];
                    rhs_vector[i] += source_function(x, y, t) * dt * dt;
                }
            }

            //solve the system using SSOR preconditioner
            dealii::SolverControl solver_control(500000, 1e-12);
            dealii::SolverCG<> solver(solver_control);
            dealii::PreconditionSSOR<> preconditioner;
            preconditioner.initialize(system_matrix, 1.2);
            solver.solve(system_matrix, solution, rhs_vector, preconditioner);

            older_solution.swap(old_solution);
            old_solution.swap(solution); //used swap in order to optimize performances 

            output_vtk(t);
        }
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

        //precompute SSOR preconditioner only once (performances)
        dealii::PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        for (double t = 0; t < T; t += dt) {
            apply_boundary_conditions(t);

            #pragma omp parallel
            {
                system_matrix.vmult(laplacian, solution);
            }

            #pragma omp parallel for simd
            for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                rhs_vector[i] += laplacian[i] * dt * dt / 2.0;
            }

            if (source_function) {
                #pragma omp parallel for simd
                for (unsigned int i = 0; i < rhs_vector.size(); ++i) {
                    double x = dof_handler.get_fe().get_unit_support_points()[i][0];
                    double y = dof_handler.get_fe().get_unit_support_points()[i][1];
                    rhs_vector[i] += source_function(x, y, t) * dt * dt / 2.0;
                }
            }

            dealii::SolverControl solver_control(500000, 1e-10);
            dealii::SolverCG<> solver(solver_control);
            solver.solve(system_matrix, solution, rhs_vector, preconditioner);

            older_solution.swap(old_solution);
            old_solution.swap(solution);

            output_vtk(t); //output of a vtk file 
        }
    } catch (const std::exception &e) {
        std::cerr << "Error during Crank-Nicolson: " << e.what() << std::endl;
        exit(1);
    }
}
//end 


//start of output function 
void WaveSolver::output_to_dx(double time) {
    const std::string dx_filename = "solution-" + std::to_string(time) + ".dx";

    //create the DataOut object
    dealii::DataOut<2> data_out;

    //attach the degree of freedom handler to the DataOut object
    data_out.attach_dof_handler(dof_handler);

    //add solution data (for example, the solution on each cell or node)
    data_out.add_data_vector(solution, "solution");

    //write the file in .dx format
    std::ofstream dx_file(dx_filename);
    data_out.write_dx(dx_file);
}
//end


//start of function that creates files vtk
void WaveSolver::output_vtk(double time) {
    constexpr unsigned int dim = 2;
    dealii::DataOut<dim> data_out;
    
    //attach the degree of freedom handler to the DataOut object
    data_out.attach_dof_handler(dof_handler);
    
    //add solution data (for example, the solution vector)
    data_out.add_data_vector(solution, "solution");
    
    //build the mesh patches for visualization
    data_out.build_patches();

    //generate the VTK file name
    std::string vtk_filename = "solution-" + std::to_string(time) + ".vtk";
    std::ofstream vtk_file(vtk_filename);

    //write the file in .vtk format
    data_out.write_vtk(vtk_file);

    //optionally, print a confirmation message to the console (too long)
    // std::cout << "VTK file generated: " << vtk_filename << std::endl;
}
//end


//calculate the error (L2 and L∞ norms)
double WaveSolver::calculate_error() {
    double error_l2 = 0.0, error_linf = 0.0;
    double dx = Lx / Nx, dy = Ly / Ny;

    //parallelize the error calculation using OpenMP
    #pragma omp parallel for reduction(+:error_l2) reduction(max:error_linf)
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
    int Nx = 50, Ny = 50;
    double Lx = 10.0, Ly = 10.0, T = 1.0, dt = 0.1;

    //loop over different grid resolutions
    for (int i = 0; i < 5; ++i) {
        //create a new solver instance for each grid resolution
        WaveSolver solver(Lx, Ly, Nx, Ny, T, dt);

        //set initial conditions for the solver
        solver.set_initial_conditions(
            [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); },
            [](double x, double y) { return 0.0; }
        );

        //solve using the Newmark method
        solver.solve_newmark();

        //calculate the error in the solution (L2 and L∞ norms)
        double error = solver.calculate_error();

        //output the error in a more readable format
        std::cout << std::fixed << std::setprecision(6);  // 6 decimal places
        std::cout << "Error for Nx = " << Nx << ", Ny = " << Ny << ": " << error << std::endl;

        //double Nx and Ny for the next iteration (refine the grid)
        Nx *= 2;
        Ny *= 2;

        //adjust the time step based on the grid resolution
        dt = 0.1 * std::min(Lx / Nx, Ly / Ny);  //adapt the time step for finer grids
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
