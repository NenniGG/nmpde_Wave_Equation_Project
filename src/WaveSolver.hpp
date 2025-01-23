#ifndef WAVESOLVER_HPP
#define WAVESOLVER_HPP

// Inclusione delle librerie necessarie da deal.II per il calcolo numerico.
#include <deal.II/base/function.h>              // Per la gestione delle funzioni matematiche
#include <deal.II/grid/tria.h>                  // Per la gestione della triangolazione
#include <deal.II/dofs/dof_handler.h>           // Per la gestione dei gradi di libertà
#include <deal.II/fe/fe_q.h>                    // Per l'elemento finito Q
#include <deal.II/lac/sparse_matrix.h>          // Per le matrici sparse
#include <deal.II/lac/vector.h>                 // Per i vettori (soluzioni, ecc.)
#include <deal.II/lac/solver_cg.h>              // Per l'uso del solver CG (Conjugate Gradient)
#include <deal.II/lac/precondition.h>           // Per l'uso del precondizionatore
#include <functional>                           // Per l'uso delle funzioni std::function

// La classe WaveSolver si occupa di risolvere l'equazione delle onde
// usando metodi numerici come Newmark e Crank-Nicolson.
class WaveSolver {
public:
    // Costruttore della classe, che definisce le dimensioni del dominio e il passo temporale.
    // Lx, Ly sono le dimensioni del dominio 2D.
    // Nx, Ny sono il numero di suddivisioni (gradi di libertà per ciascuna direzione).
    // T è il tempo finale, mentre dt è il passo temporale.
    WaveSolver(double Lx, double Ly, int Nx, int Ny, double T, double dt);

    // Funzione per impostare le condizioni iniziali di u0 (posizione iniziale) e u1 (velocità iniziale).
    // Le funzioni sono passate come oggetti std::function che accettano due variabili (x, y).
    void set_initial_conditions(const std::function<double(double, double)> &u0,
                                const std::function<double(double, double)> &u1);

    // Funzione per impostare le condizioni al contorno, in funzione del tempo.
    // La funzione g accetta le coordinate (x, y) e il tempo t, ed è definita come std::function.
    void set_boundary_condition(const std::function<double(double, double, double)> &g);

    // Funzione per impostare il termine sorgente f, che dipende dalle coordinate spaziali (x, y) e dal tempo t.
    // Anch'essa è una funzione definita tramite std::function.
    void set_source_function(const std::function<double(double, double, double)> &f);

    // Funzione per risolvere l'equazione usando il metodo di Newmark.
    void solve_newmark();

    // Funzione per risolvere l'equazione usando il metodo di Crank-Nicolson.
    void solve_crank_nicolson();

    // Funzione per analizzare le prestazioni del solver (es. il tempo di calcolo).
    void analyze_performance() const;

    // Funzione per calcolare l'errore tra la soluzione numerica e la soluzione esatta.
    // Restituisce un valore di errore.
    double calculate_error(); // Modifica della dichiarazione

    // Funzione per testare la convergenza del metodo numerico (convergenza in funzione del passo spaziale o temporale).
    void test_convergence();  // Dichiarazione della funzione

private:
    // Funzione per impostare il sistema lineare (matrici, vettori, ecc.) per la soluzione numerica.
    void setup_system();

    // Funzione per applicare le condizioni al contorno su ciascun passo temporale.
    void apply_boundary_conditions(double time);

    // Funzione per ripristinare le soluzioni a valori iniziali, prima di un nuovo passo temporale.
    void reset_solutions();

    // Parametri principali della simulazione:
    const double Lx, Ly;  // Dimensioni del dominio in 2D.
    const int Nx, Ny;      // Numero di gradi di libertà nelle due direzioni spaziali.
    const double T, dt;    // Tempo finale della simulazione e passo temporale.

    // Gli oggetti deal.II necessari per rappresentare il dominio e le soluzioni.
    dealii::Triangulation<2> triangulation;  // Rappresentazione della triangolazione 2D del dominio.
    dealii::FE_Q<2> fe;                     // Funzione di base degli elementi finiti (ordine 2).
    dealii::DoFHandler<2> dof_handler;      // Gestore dei gradi di libertà per il sistema 2D.

    // Matrici e vettori per il sistema lineare, contenenti le soluzioni e i termini sorgente.
    dealii::SparsityPattern sparsity_pattern;   // Modello della struttura sparsa della matrice.
    dealii::SparseMatrix<double> system_matrix; // La matrice del sistema lineare.
    
    // Vettori per contenere le soluzioni ai vari tempi e il vettore del termine destro (rhs).
    dealii::Vector<double> solution;           // Vettore che contiene la soluzione corrente.
    dealii::Vector<double> old_solution;      // Soluzione al passo temporale precedente.
    dealii::Vector<double> older_solution;    // Soluzione al passo temporale antecedente.
    dealii::Vector<double> rhs_vector;        // Vettore del termine destro del sistema lineare.

    // Funzioni sorgente e al contorno, definite come funzioni generiche.
    std::function<double(double, double, double)> boundary_condition;  // Condizioni al contorno.
    std::function<double(double, double, double)> source_function;     // Funzione sorgente.
};

#endif // WAVESOLVER_HPP
