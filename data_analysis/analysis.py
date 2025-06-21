"""
Module for analyzing exponential decay data with multiple components.

This module provides functions for fitting multiple exponential decay components
to experimental data using a sequential fitting approach.
"""
# %% Import libraries
import os
import sys

# Add parallel_iir_filters directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filters_path = os.path.join(project_root, 'arch', 'poc', 'parallel_iir_filters')
if filters_path not in sys.path:
    sys.path.append(filters_path)

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
from scipy.optimize import curve_fit
import json

# Import filters_wrapper from src
from src.filters_wrapper import filters_wrapper
from src.config.filter_cfg import FilterCfg

# %% Define functions
def single_exp_decay(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    """Single exponential decay without offset
    
    Args:
        t (array): Time points
        amp (float): Amplitude of the decay
        tau (float): Time constant of the decay
        
    Returns:
        array: Exponential decay values
    """
    return amp * np.exp(-t/tau)

def sequential_exp_fit(
    t: np.ndarray, 
    y: np.ndarray, 
    start_fractions: list[float], 
    verbose: bool=True
    ) -> tuple[list[tuple[float, float]], float, np.ndarray]:
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term from the tail of the data
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        start_fractions (list): List of fractions (0 to 1) indicating where to start fitting each component
        verbose (bool): Whether to print detailed fitting information
        
    Returns:
        tuple: (components, a_dc, residual) where:
            - components: List of (amplitude, tau) pairs for each fitted component
            - a_dc: Fitted constant term
            - residual: Residual after subtracting all components
    """
    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0
    
    # First, estimate the constant term from the tail of the data
    # Find the flat region in the tail by looking at local variance
    window = len(y) // 100  # Window size by dividing signal into 100 equal pieces
    rolling_var = np.array([np.var(y[i:i+window]) for i in range(len(y)-window)])
    # Find where variance drops below threshold, indicating flat region
    var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
    flat_start = np.where(rolling_var < var_threshold)[0][-1]
    a_dc = np.mean(y[flat_start:])
    if verbose:
        print(f"\nFitted constant term: {a_dc:.3e}")
    
    y_residual = y.copy() - a_dc
    
    for i, start_frac in enumerate(start_fractions):
        # Calculate start index for this component
        start_idx = int(len(t) * start_frac)
        if verbose:
            print(f"\nFitting component {i+1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")
        
        # Fit current component
        try:
            # Initial guess for parameters
            p0 = [
                y_residual[start_idx],  # amplitude
                t_offset[start_idx] / 3  # tau
            ]
            
            # Set bounds for the fit
            bounds = (
                [-np.inf, 0],  # lower bounds: amplitude can be negative, tau must be positive
                [np.inf, np.inf]  # upper bounds
            )
            
            # Perform the fit on the current interval
            t_fit = t_offset[start_idx:]
            y_fit = y_residual[start_idx:]
            popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)
            
            # Store the components
            amp, tau = popt
            components.append((amp, tau))
            if verbose:
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns")
            
            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset/tau)
            
        except RuntimeError as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i+1}: {e}")
            break
    
    return components, a_dc, y_residual

def optimize_start_fractions_filtered(t, y, base_fractions, bounds_scale=0.5):
    """
    Optimize the start_fractions by minimizing the RMS between filtered signal and ideal step
    using scipy.optimize.minimize.
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        base_fractions (list): Initial guess for start fractions
        bounds_scale (float): Scale factor for bounds around base fractions (0.5 means ±50%)
        
    Returns:
        tuple: (best_fractions, best_components, best_dc, best_rms_filtered)
    """
    from scipy.optimize import minimize
    
    def objective(x):
        """Objective function to minimize: RMS between filtered signal and ideal step"""
        # Ensure fractions are ordered (f1 > f2 > f3)
        if not (x[0] > x[1] > x[2]):
            return 1e6  # Return large value if constraint is violated
        
        try:
            # Try this combination of fractions
            components, a_dc, _ = sequential_exp_fit(t, y, x, verbose=False)
            
            # Calculate filtered signal and its RMS
            A = [component[0] for component in components]
            tau = [component[1]*1e-9 for component in components]
            y_scaled = y * 0.4  # Scale the signal
            y_filtered = filters_wrapper(y_scaled, A, tau, a_dc)
            step = 0.4 * np.ones_like(y_filtered)
            filtered_residual = step - y_filtered
            current_rms = np.sqrt(np.mean(filtered_residual**2))
            
            return current_rms
        
        except RuntimeError:
            return 1e6  # Return large value if fit fails
    
    # Define bounds for optimization
    bounds = []
    for base in base_fractions:
        min_val = base * (1 - bounds_scale)
        max_val = base * (1 + bounds_scale)
        bounds.append((min_val, max_val))
    
    print("\nOptimizing start_fractions using scipy.optimize.minimize...")
    print(f"Initial values: {[f'{f:.5f}' for f in base_fractions]}")
    print(f"Bounds: ±{bounds_scale*100}% around initial values")
    
    # Run optimization
    result = minimize(
        objective,
        x0=base_fractions,
        bounds=bounds,
        method='Nelder-Mead',  # This method works well for non-smooth functions
        options={'disp': True, 'maxiter': 200}
    )
    
    # Get final results
    if result.success:
        best_fractions = result.x
        components, a_dc, _ = sequential_exp_fit(t, y, best_fractions, verbose=False)
        
        # Calculate final RMS
        A = [component[0] for component in components]
        tau = [component[1]*1e-9 for component in components]
        y_scaled = y * 0.4
        y_filtered = filters_wrapper(y_scaled, A, tau, a_dc)
        step = 0.4 * np.ones_like(y_filtered)
        filtered_residual = step - y_filtered
        best_rms_filtered = np.sqrt(np.mean(filtered_residual**2))
        
        print("\nOptimization successful!")
        print(f"Initial fractions: {[f'{f:.5f}' for f in base_fractions]}")
        print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
        print(f"Final RMS filtered: {best_rms_filtered:.3e}")
        print(f"Number of iterations: {result.nit}")
    else:
        print("\nOptimization failed. Using initial values.")
        best_fractions = base_fractions
        components, a_dc, _ = sequential_exp_fit(t, y, best_fractions)
        best_rms_filtered = objective(best_fractions)
    
    return best_fractions, components, a_dc, best_rms_filtered

# %% Load and prepare data

if __name__ == '__main__':
    # file_path = "/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/OPX1000_LF_out_step_response_2GSaps/normalized_amplitude_ch6.npy"
    # normalized_amplitude = np.load(file_path)
    # y = normalized_amplitude[750:5000]  # Use the same data range as original plot
    # t = np.arange(len(y)) * 0.5  # time in ns (0.5ns is the sampling period)
    
    file_path1 = "/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/Cryoscope_data/time_1d_4144.npy"
    file_path2 = "/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/Cryoscope_data/flux_ampl_4144.npy"
    t = np.load(file_path1)
    y = np.load(file_path2)

    # Define base start fractions and optimize them
    base_fractions = [0.1, 0.013, 0.005]
    best_fractions, best_components, best_a_dc, best_rms_filtered = optimize_start_fractions_filtered(t, y, base_fractions, bounds_scale=0)

    # Use the optimized results for plotting
    components = best_components
    a_dc = best_a_dc

# %% Plot results
plt.figure(figsize=(12, 8))

# Set font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Define a custom color palette
colors = ['#1f77b4',  # blue
          '#2ca02c',  # green
          '#ff7f0e',  # orange
          '#d62728',  # red
          '#9467bd',  # purple
          '#8c564b',  # brown
          '#e377c2',  # pink
          '#7f7f7f',  # gray
          '#bcbd22',  # yellow-green
          '#17becf'   # cyan
]

# Plot original data
plt.semilogx(t, y, color=colors[0], marker='.', linestyle='None', label='Original Data', alpha=0.7)

# Generate and plot full fit
t_offset = t - t[0]
y_fit = np.ones_like(t, dtype=float) * a_dc  # Start with fitted constant
plt.axhline(y=a_dc, color=colors[1], linestyle='--', label=f'A$_{{\mathrm{{dc}}}}$ = {a_dc:.3e}', alpha=0.5)

for i, (amp, tau) in enumerate(components):
    # Plot individual components
    component = amp * np.exp(-t_offset/tau) + a_dc
    plt.plot(t, component, '--', color=colors[i+2],
            label=f'Component {i+1} (A = {amp:.3e}, τ = {tau:.3f} ns, start={t[int(len(t)*best_fractions[i])]:.1f} ns)', alpha=0.5)
    y_fit += amp * np.exp(-t_offset/tau)

# Plot full fit
plt.plot(t, y_fit, color=colors[3], label='Full Fit', linewidth=2)  # Red

# Plot residual
plt.plot(t, y - y_fit + best_a_dc, color='k', linestyle=':', label='Residual') # Black 

plt.grid(True)
plt.legend()

# Set axis labels with larger font
plt.xlabel('Time (ns)')
plt.ylabel('Normalized Amplitude')

# Increase tick label sizes
plt.xticks()
plt.yticks()

# Print fitted parameters
print("\nFitted Parameters:")
print(f"Constant term: {a_dc:.3e}")
for i, (amp, tau) in enumerate(components):
    print(f"\nComponent {i+1} (start time: {t[int(len(t)*best_fractions[i])]:.1f} ns):")
    print(f"Amplitude: {amp:.3e}")
    print(f"Time Constant: {tau:.3f} ns")
    print(f"Start fraction: {best_fractions[i]:.5f}")

print(f"\nFinal RMS residual: {np.sqrt(np.mean((y - y_fit)**2)):.3e}")

plt.title('Sequential Multi-Exponential Fit', pad=20)


# %% Plot the original and filtered signal in a new figure
step_factor = 1

y *= step_factor
plt.figure(figsize=(12, 8))

A = [component[0] for component in components]
tau = [component[1]*1e-9 for component in components]

cfg = FilterCfg(verbose=False)
y_filtered = filters_wrapper(y, A, tau, a_dc, cfg=cfg)

step = step_factor * np.ones_like(y_filtered)
fir = scipy.signal.deconvolve(step, y_filtered[:100])[0][0:47]
fir[-1] += 1 - np.sum(fir)

y_filtered2 = filters_wrapper(y, A, tau, a_dc, fir=fir, cfg=cfg)

plt.semilogx(t, y / step_factor, color=colors[0], label='Original Data', alpha=0.7)
plt.plot(t, y_filtered / step_factor, color=colors[3], label='Filtered Signal with IIR', linewidth=2)
plt.plot(t, y_filtered2 / step_factor, color=colors[4], label='Filtered Signal with IIR+FIR', linewidth=2)
plt.plot(t, step / step_factor, color=colors[1], linestyle='-', linewidth=2, label='Ideal step')

filtered_residual = step - y_filtered
rms_filtered = np.sqrt(np.mean(filtered_residual**2))
print(f"RMS filtered: {rms_filtered:.3e}")

plt.grid(True)
plt.legend()

# Set axis labels with larger font
plt.xlabel('Time (ns)')
plt.ylabel('Normalized Amplitude')

# Increase tick label sizes
plt.xticks()
plt.yticks()

plt.title('Original vs Filtered Signal', pad=20)
plt.show()

# %% Save fitting parameters to JSON

# Create a dictionary with the fitting parameters
fit_params = {
    'a_dc': float(a_dc),  # Convert numpy float to Python float for JSON serialization
    'exponentials': [
        {
            'amplitude': float(amp),
            'tau': float(tau)
        }
        for (amp, tau) in components
    ]
}

# Save to JSON file
output_path = "/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/fit_parameters.json"
with open(output_path, 'w') as f:
    json.dump(fit_params, f, indent=4)

print(f"\nFitting parameters saved to: {output_path}")
