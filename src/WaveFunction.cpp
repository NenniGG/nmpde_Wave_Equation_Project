// start libraries 
#include "WaveFunction.hpp"
//end 
//================================================================================================================================
namespace WaveEq
{
  using namespace dealii;

  template <int dim>
  WaveEquation<dim>::WaveEquation()
    : fe(1)                                       // finite element of degree 1
    , dof_handler(triangulation)                 // initialize the degree of freedom handler with the mesh
    , time_step(1. / 64)                         // time step size
    , time(time_step)                            // current simulation time (starts at one time step)
    , timestep_number(1)                         // counter for time steps
    , theta(0.5)                                 // theta method parameter (0.5 = Crank-Nicolson)
  {}
//================================================================================================================================

  //start setup function 
  template <int dim>
  void WaveEquation<dim>::setup_system()
  {
    // create a 2D mesh from -1 to 1 in each direction and refine it uniformly 7 times
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(7);

    // distribute degrees of freedom for the finite element
    dof_handler.distribute_dofs(fe);

    // create a sparsity pattern based on the dof_handler
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // initialize system matrices with the sparsity pattern
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    // create mass and laplace matrices using Gaussian quadrature
    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    // initialize solution and system vectors
    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close(); // close constraints (no constraints applied here)
  }
  //end setup function 
//================================================================================================================================

  // start solve function for u
  template <int dim>
  void WaveEquation<dim>::solve_u()
  {
    // solve the linear system for solution_u using the conjugate gradient method
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());
  }
  //end 
//================================================================================================================================
  // start solve function for velocity 
  template <int dim>
  void WaveEquation<dim>::solve_v()
  {
    // solve the linear system for solution_v using the conjugate gradient method
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());
  }
  //end 
//================================================================================================================================
  //start function for generating output
  template <int dim>
  void WaveEquation<dim>::output_results() const
  {
    // create VTK output file with solution data
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(solution_v, "V");
    data_out.build_patches();

    // create the output filename based on the current timestep
    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    const std::string full_filename = "output/" + filename;

    // set compression level for faster writing
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    // write the VTU file
    std::ofstream output(full_filename);
    data_out.write_vtu(output);

    // update the PVD file for time-series visualization in ParaView
    static std::ofstream pvd_file("output/solution.pvd", std::ios::app);
    static bool first_time = true;

    if (first_time)
    {
      pvd_file << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
               << "  <Collection>\n";
      first_time = false;
    }

    // write a reference to the new .vtu file with its associated timestep
    pvd_file << "    <DataSet timestep=\"" << time
             << "\" group=\"\" part=\"0\" file=\"" << filename << "\"/>\n";

    // close the PVD file at the end of the simulation
    if (time + time_step > 5.0)
    {
      pvd_file << "  </Collection>\n"
               << "</VTKFile>\n";
      pvd_file.close();
    }
  }
  //end 
//================================================================================================================================
  //start run function 
  template <int dim>
  void WaveEquation<dim>::run()
  {
    // initialize system matrices and vectors
    setup_system();

    // project initial conditions into the finite element space
    VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                         InitialValuesU<dim>(), old_solution_u);
    VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                         InitialValuesV<dim>(), old_solution_v);

    Vector<double> tmp(solution_u.size());
    Vector<double> forcing_terms(solution_u.size());

    // time-stepping loop from t = 0 to t = 5
    for (; time <= 5; time += time_step, ++timestep_number)
    {
      // update right-hand side using previous solutions
      mass_matrix.vmult(system_rhs, old_solution_u);
      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs.add(time_step, tmp);

      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

      // compute forcing term at current and previous time steps
      RightHandSide<dim> rhs;
      rhs.set_time(time);
      VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs, tmp);
      forcing_terms = tmp;
      forcing_terms *= theta * time_step;

      rhs.set_time(time - time_step);
      VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs, tmp);
      forcing_terms.add((1 - theta) * time_step, tmp);
      system_rhs.add(theta * time_step, forcing_terms);

      // apply boundary conditions and solve for solution_u
      {
        BoundaryValuesU<dim> bvu;
        bvu.set_time(time);
        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dof_handler, 0, bvu, bv);
        matrix_u.copy_from(mass_matrix);
        matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
        MatrixTools::apply_boundary_values(bv, matrix_u, solution_u, system_rhs);
      }
      solve_u();

      // compute system_rhs for solution_v and solve
      laplace_matrix.vmult(system_rhs, solution_u);
      system_rhs *= -theta * time_step;
      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs += tmp;
      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-time_step * (1 - theta), tmp);
      system_rhs += forcing_terms;

      // apply boundary conditions and solve for solution_v
      {
        BoundaryValuesV<dim> bvv;
        bvv.set_time(time);
        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dof_handler, 0, bvv, bv);
        matrix_v.copy_from(mass_matrix);
        MatrixTools::apply_boundary_values(bv, matrix_v, solution_v, system_rhs);
      }
      solve_v();

      // output current results and update old solutions
      output_results();
      old_solution_u = solution_u;
      old_solution_v = solution_v;
    }
  }
  //end 
//================================================================================================================================
  // equation data implementations
  //start 
  template <int dim>
  double InitialValuesU<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    const double r2 = p.square();     // squared distance from origin
    return std::exp(-100 * r2);       // initial peak at center (gaussian)
  }

  template <int dim>
  double InitialValuesV<dim>::value(const Point<dim> &, const unsigned int) const
  {
    return 0.0;                       // initial velocity is zero everywhere
  }

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
  {
    return 0;                         // no external force applied
  }

  template <int dim>
  double BoundaryValuesU<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return 0.0;
  }

  template <int dim>
  double BoundaryValuesV<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return 0.0;
  }

  // explicit template instantiation for 2D problem
  template class WaveEquation<2>;
  template class InitialValuesU<2>;
  template class InitialValuesV<2>;
  template class RightHandSide<2>;
  template class BoundaryValuesU<2>;
  template class BoundaryValuesV<2>;

} // namespace WaveEq
//end 
