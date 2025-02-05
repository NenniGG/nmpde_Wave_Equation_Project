#ifndef WAVESOLVER_HPP
#define WAVESOLVER_HPP


//include the necessary deal.II libraries for numerical computation
#include <deal.II/base/function.h>              //for handling mathematical functions
#include <deal.II/grid/tria.h>                  //for handling triangulation
#include <deal.II/dofs/dof_handler.h>           //for handling degrees of freedom
#include <deal.II/fe/fe_q.h>                    //for finite element Q element
#include <deal.II/lac/sparse_matrix.h>          //for sparse matrices
#include <deal.II/lac/vector.h>                 //for vectors (solutions, etc.)
#include <deal.II/lac/solver_cg.h>              //for using the CG (Conjugate Gradient) solver
#include <deal.II/lac/precondition.h>           //for using preconditioners
#include <functional>                           //for using std::function objects
//end libraries 

using namespace dealii;



// Define a class to represent the exact solution for computing the error
class ExactSolution : public Function<2> {
public:
    virtual double value(const Point<2> &p, const unsigned int /*component*/ = 0) const override {
        return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);  // Example analytical solution
    }
};



//the WaveSolver class is responsible for solving the wave equation using numerical methods such as Crank-Nicolson
class WaveSolver {
public:
    //constructor of the class, which defines the dimensions of the domain and the time step. Lx, Ly are the dimensions of the 2D domain
    //Nx, Ny are the number of subdivisions (degrees of freedom for each direction)
    //T is the final time, while dt is the time step
    WaveSolver(double Lx, double Ly, int Nx, int Ny, double T, double dt);


    // Dichiarazione della funzione che verifica NaN e infini nei vettori
    void check_for_nan_in_vector(dealii::Vector<double>& vec);

    // Dichiarazione della funzione che verifica la matrice di sistema
   // Dichiarazione della funzione che verifica la matrice di sistema
void check_for_empty_matrix(const dealii::SparseMatrix<double>& matrix);


    //function to set the initial conditions for u0 (initial position) and u1 (initial velocity)
    //the functions are passed as std::function objects that accept two variables (x, y)
    void set_initial_conditions(const std::function<double(double, double)> &u0,
                                const std::function<double(double, double)> &u1);

    //function to set the boundary conditions, depending on time
    //the function g accepts the coordinates (x, y) and the time t, and is defined as a std::function
    void set_boundary_condition(const std::function<double(double, double, double)> &g);

    //function to set the source term f, which depends on the spatial coordinates (x, y) and time t
    //this is also a function defined via std::function
    void set_source_function(const std::function<double(double, double, double)> &f);

    void output_vtk(double time);

    void output_to_dx(double time);

    //function to solve the equation using the Crank-Nicolson method
    void solve_crank_nicolson();

    //function to analyze the performance of the solver (e.g., computation time)
    void analyze_performance() const;
     double compute_error() const;  // Computes the error (L2 norm) of the solution
      //function to apply boundary conditions at each time step
    void apply_boundary_conditions(double time);


private:
    dealii::Vector<double> system_rhs; //declaration of the RHS vector
    std::map<unsigned int, double> boundary_values; // Assumendo che sia una mappa di valori al cont

    //function to set up the linear system (matrices, vectors, etc.) for the numerical solution
    void setup_system();

    //support function that builds system matrix
    void assemble_system_matrix();

    bool solve_linear_system(dealii::SparseMatrix<double>& matrix, dealii::Vector<double>& sol, const dealii::Vector<double>& rhs);

   

    //function to reset the solutions to initial values before a new time step
    void reset_solutions();


    //main simulation parameters:
    const double Lx, Ly;  //dimensions of the 2D domain
    const int Nx, Ny;      //number of degrees of freedom in the two spatial directions
    const double T, dt;    //final time of the simulation and time step

    //deal.II objects needed to represent the domain and the solutions
    dealii::Triangulation<2> triangulation;  //representation of the 2D domain triangulation
    dealii::FE_Q<2> fe;                     //finite element basis (order 2)
    dealii::DoFHandler<2> dof_handler;      //degree of freedom handler for the 2D system

    //matrices and vectors for the linear system, containing the solutions and source terms
    dealii::SparsityPattern sparsity_pattern;   //pattern of the sparse matrix structure
    dealii::SparseMatrix<double> system_matrix; //the linear system matrix
    
    //vectors to hold the solutions at different time steps and the right-hand side (rhs) vector
    dealii::Vector<double> solution;           //vector holding the current solution
    dealii::Vector<double> old_solution;      //solution at the previous time step
    dealii::Vector<double> older_solution;    //solution at the time step before the previous
    dealii::Vector<double> rhs_vector;        //right-hand side vector for the linear system

    //source and boundary functions, defined as generic functions
    std::function<double(double, double, double)> boundary_condition;  //boundary conditions
    std::function<double(double, double, double)> source_function;     //source function
};

#endif // WAVESOLVER_HPP
