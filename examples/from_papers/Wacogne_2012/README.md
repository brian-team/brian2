# MMN Large Scale Simulation

This project implements a large-scale Mismatch Negativity (MMN) simulation using Brian2. It models cortical columns and memory traces to investigate deviance detection mechanisms.

## Project Structure

The project has been modularized for better maintainability and readability:

*   **`main.py`**: The entry point of the simulation. Used to configure parameters and launch experiments.
*   **`src/`**: Source code directory.
    *   **`network.py`**: Contains functions to build neuron groups, synapses, and cortical columns.
    *   **`simulation.py`**: Core logic for running simulations, including paradigm generation (Classic, Alternating, etc.).
    *   **`analysis.py`**: Functions for analyzing spike data, detecting omission responses, and calculating statistics.
    *   **`plotting.py`**: Visualization tools for generating raster plots, PSTHs, and weight profile figures.

## Installation

Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: This project requires `brian2`, `numpy`, and `matplotlib`.*

## Usage

To run the simulation:

```bash
python main.py
```

### Configuration

You can select the experiment type in `main.py` by changing the `experiment_to_run` variable:

*   `'classic'`
*   `'alternating'`
*   `'local_global'`
*   `'omission'`
*   `'figure4_multi'` (Reproduces Figure 4 from the reference paper)

Output figures are saved in the `fig_out/` directory.

## Contributors

*   AtakanDogan21 (https://github.com/AtakanDogan21)
