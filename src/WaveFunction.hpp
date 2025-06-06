#ifndef WAVE_EQUATION_HPP
#define WAVE_EQUATION_HPP

// include standard deal.II headers for mathematical functions, integration rules, and utilities
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

// include deal.II linear algebra components
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

// include deal.II mesh generation and handling
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

// include degree of freedom tools
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// include finite element class
#include <deal.II/fe/fe_q.h>

// include deal.II numerical postprocessing tools
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// include standard C++ headers for file I/O and error output
#include <fstream>
#include <iostream>
//================================================================================================================================
//start
namespace WaveEq
{
  using namespace dealii;

  // main wave equation solver class
  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation();  // constructor
    void run();      // main function to run the simulation

  private:
    // internal helper functions
    void setup_system();      // create mesh, dofs, and initialize matrices
    void solve_u();           // solve for displacement u
    void solve_v();           // solve for velocity v
    void output_results() const; // write results to VTU/PVD files

    // core simulation components
    Triangulation<dim> triangulation;  // mesh
    const FE_Q<dim>    fe;             // finite element
    DoFHandler<dim>    dof_handler;    // manages degrees of freedom

    AffineConstraints<double> constraints; // constraints for Dirichlet BCs (currently empty)

    // matrices used in the time integration
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    // vectors for current and previous solutions
    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    // time-stepping variables
    double       time_step;       // delta t
    double       time;            // current simulation time
    unsigned int timestep_number; // current timestep index
    const double theta;           // theta parameter for time integration (0.5 = Crank-Nicolson)
  };

  // class defining the initial condition for displacement u
  template <int dim>
  class InitialValuesU : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  // class defining the initial condition for velocity v
  template <int dim>
  class InitialValuesV : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  // class for defining the right-hand side forcing term f
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  // class defining the Dirichlet boundary condition for u
  template <int dim>
  class BoundaryValuesU : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  // class defining the Dirichlet boundary condition for v
  template <int dim>
  class BoundaryValuesV : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

} // namespace WaveEq

#endif // WAVE_EQUATION_HPP
//end 