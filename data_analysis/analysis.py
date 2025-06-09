# %% Import libraries
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from .arch.poc.parallel_iir_filters.src.filters_wrapper import filters_wrapper

# %% Define functions
def single_exp_decay(t, amp, tau):
    """Single exponential decay without offset"""
    return amp * np.exp(-t/tau)

def sequential_exp_fit(t, y, start_fractions):
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components
    
    Parameters:
    -----------
    t : array
        Time points in nanoseconds
    y : array
        Data points
    start_fractions : list
        List of fractions (0 to 1) indicating where to start fitting each component
    """
    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0
    
    # First, estimate the constant term from the tail of the data
    constant = np.mean(y[-1000:])  # Use last 20 points
    print(f"\nFitted constant term: {constant:.3e}")
    
    y_residual = y.copy() - constant
    
    for i, start_frac in enumerate(start_fractions):
        # Calculate start index for this component
        start_idx = int(len(t) * start_frac)
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
            print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns")
            
            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset/tau)
            
        except RuntimeError as e:
            print(f"Warning: Fitting failed for component {i+1}: {e}")
            break
    
    return components, constant, y_residual

# %%
# Load and prepare data
file_path = "/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/OPX1000_LF_out_step_response_2GSaps/normalized_amplitude_ch6.npy"
normalized_amplitude = np.load(file_path)
y = normalized_amplitude[750:5000]  # Use the same data range as original plot
t = np.arange(len(y)) * 0.5  # time in ns (0.5ns is the sampling period)

# Define start fractions for each component
# Adjust these values to control fitting ranges
start_fractions = [0.1, 0.015, 0.005]  # Example: 5 components with different ranges

# Perform sequential fitting
components, constant, residual = sequential_exp_fit(t, y, start_fractions)
print(f"RMS residual: {np.sqrt(np.mean(residual**2)):.3e}")

# %%
# Plot results
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

# Plot original data
ax.semilogx(t, y, 'b.', label='Original Data', alpha=0.7)

# Generate and plot full fit
t_offset = t - t[0]
y_fit = np.ones_like(t, dtype=float) * constant  # Start with fitted constant
ax.axhline(y=constant, color='g', linestyle='--', label=f'Constant = {constant:.3e}', alpha=0.5)

for i, (amp, tau) in enumerate(components):
    # Plot individual components
    component = amp * np.exp(-t_offset/tau) + constant
    ax.plot(t, component, '--', 
            label=f'Component {i+1} (A = {amp:.3e}, τ = {tau:.3f} ns, start={t[int(len(t)*start_fractions[i])]:.1f} ns)', alpha=0.5)
    y_fit += amp * np.exp(-t_offset/tau)

# Plot full fit
ax.plot(t, y_fit, 'r-', label='Full Fit', linewidth=2)

# Plot residual
ax.plot(t, residual + constant, 'k:', label='Residual', alpha=0.5)

ax.grid(True)
ax.legend()

# Set axis labels
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Normalized Amplitude')

# Print fitted parameters
print("\nFitted Parameters:")
print(f"Constant term: {constant:.3e}")
for i, (amp, tau) in enumerate(components):
    print(f"\nComponent {i+1} (start time: {t[int(len(t)*start_fractions[i])]:.1f} ns):")
    print(f"Amplitude: {amp:.3e}")
    print(f"Time Constant: {tau:.3f} ns")
    print(f"Start fraction: {start_fractions[i]:.5f}")

print(f"\nFinal RMS residual: {np.sqrt(np.mean(residual**2)):.3e}")

plt.title('Sequential Multi-Exponential Fit with Fitted Constant', pad=20)
plt.show()

# Save fitting parameters to JSON

# Create a dictionary with the fitting parameters
fit_params = {
    'constant': float(constant),  # Convert numpy float to Python float for JSON serialization
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
