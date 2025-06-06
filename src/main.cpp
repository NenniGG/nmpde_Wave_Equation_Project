//start libraries 
#include "WaveFunction.hpp"             // include the header file that defines the WaveEquation class and related functionality
#include <filesystem>                   // required to use filesystem utilities such as directory creation
//end libraries          

namespace fs = std::filesystem;         // define a shorter alias 'fs' for 'std::filesystem' to improve code readability

//start main function 
int main()
{
  try
  {
    // create an output directory where simulation results will be saved
    // if the directory already exists, this does nothing
    fs::create_directory("output");

    // instantiate a wave solver object for the 2D wave equation
    // 'WaveEq::WaveEquation<2>' indicates a template instantiation for 2D
    WaveEq::WaveEquation<2> wave_solver;

    // run the wave equation solver
    wave_solver.run();
  }
  catch (std::exception &exc)
  {
    // catch and handle standard exceptions
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Exception on processing: " << exc.what() << "\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;  // return error code 
  }
  catch (...)
  {
    // catch and handle all other unexpected exceptions
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Unknown exception!\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;  // return error code
  }

  // return 0 to indicate successful execution
  return 0;
}
//end main 
