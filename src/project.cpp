#include "WaveSolver.hpp"
#include <filesystem> //Necessario per fs::create_directory
namespace fs = std::filesystem;

int main()
{
  try
  {
    fs::create_directory("output"); // Crea la cartella output

    WaveEq::WaveEquation<2> wave_solver;
    wave_solver.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Exception on processing: " << exc.what() << "\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Unknown exception!\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }

  return 0;
}
