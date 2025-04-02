#include "WaveSolver.hpp"

namespace WaveEq
{
  using namespace dealii;

  template <int dim>
  WaveEquation<dim>::WaveEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 64)
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
  {}

  template <int dim>
  void WaveEquation<dim>::setup_system()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(7);

    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close();
  }

  template <int dim>
  void WaveEquation<dim>::solve_u()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());
  }

  template <int dim>
  void WaveEquation<dim>::solve_v()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());
  }

  template <int dim>
    void WaveEquation<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "U");
  data_out.add_data_vector(solution_v, "V");
  data_out.build_patches();

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
  const std::string full_filename = "output/" + filename;

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);

  std::ofstream output(full_filename);
  data_out.write_vtu(output);

  // Aggiorna PVD (file master per la serie temporale)
  static std::ofstream pvd_file("output/solution.pvd", std::ios::app);
  static bool first_time = true;

  if (first_time)
  {
    pvd_file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
             << "  <Collection>\n";
    first_time = false;
  }

  pvd_file << "    <DataSet timestep=\"" << time
           << "\" group=\"\" part=\"0\" file=\"" << filename << "\"/>\n";

  // Chiudi il file PVD alla fine della simulazione
  if (time + time_step > 5.0)
  {
    pvd_file << "  </Collection>\n"
             << "</VTKFile>\n";
    pvd_file.close();
  }
}

  template <int dim>
  void WaveEquation<dim>::run()
  {
    setup_system();

    VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                         InitialValuesU<dim>(), old_solution_u);
    VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                         InitialValuesV<dim>(), old_solution_v);

    Vector<double> tmp(solution_u.size());
    Vector<double> forcing_terms(solution_u.size());

    for (; time <= 5; time += time_step, ++timestep_number)
    {
      mass_matrix.vmult(system_rhs, old_solution_u);
      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs.add(time_step, tmp);

      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

      RightHandSide<dim> rhs;
      rhs.set_time(time);
      VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs, tmp);
      forcing_terms = tmp;
      forcing_terms *= theta * time_step;

      rhs.set_time(time - time_step);
      VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs, tmp);
      forcing_terms.add((1 - theta) * time_step, tmp);
      system_rhs.add(theta * time_step, forcing_terms);

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

      laplace_matrix.vmult(system_rhs, solution_u);
      system_rhs *= -theta * time_step;
      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs += tmp;
      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-time_step * (1 - theta), tmp);
      system_rhs += forcing_terms;

      {
        BoundaryValuesV<dim> bvv;
        bvv.set_time(time);
        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dof_handler, 0, bvv, bv);
        matrix_v.copy_from(mass_matrix);
        MatrixTools::apply_boundary_values(bv, matrix_v, solution_v, system_rhs);
      }
      solve_v();

      output_results();
      old_solution_u = solution_u;
      old_solution_v = solution_v;
    }
  }

  // Equation Data Implementations
  template <int dim>
  double InitialValuesU<dim>::value(const Point<dim> &, const unsigned int) const
  {
    return 0;
  }

  template <int dim>
  double InitialValuesV<dim>::value(const Point<dim> &, const unsigned int) const
  {
    return 0;
  }

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
  {
    return 0;
  }

  template <int dim>
  double BoundaryValuesU<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) && (p[1] > -1. / 3))
      return std::sin(this->get_time() * 4 * numbers::PI);
    else
      return 0;
  }

  template <int dim>
  double BoundaryValuesV<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) && (p[1] > -1. / 3))
      return std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI;
    else
      return 0;
  }

  // Explicit template instantiation
  template class WaveEquation<2>;
  template class InitialValuesU<2>;
  template class InitialValuesV<2>;
  template class RightHandSide<2>;
  template class BoundaryValuesU<2>;
  template class BoundaryValuesV<2>;

} // namespace WaveEq
