import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filters_path = os.path.join(project_root, 'arch', 'poc', 'parallel_iir_filters')
if filters_path not in sys.path:
    sys.path.append(filters_path)

from src.filters_wrapper import filters_wrapper
from src.config.filter_cfg import FilterCfg


with open("/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/fit_parameters.json", "r") as f:
    fit_params = json.load(f)

t = np.load("/home/shlomimatit/Projects/Qolab/Qolab_projects/data_analysis/t_array.npy")

# Remove duplicates from t array while preserving order
unique_indices = np.unique(t, return_index=True)[1]
unique_indices = np.sort(unique_indices)
t_unique = t[unique_indices]

step_factor = 0.4

a_dc = fit_params['a_dc']
components = fit_params['exponentials']

y_fit = np.ones_like(t_unique, dtype=float) * a_dc  # Start with fitted constant

for component in components:
    y_fit += component['amplitude'] * np.exp(-(t_unique-t_unique[0])/component['tau'])

y_fit *= step_factor

# Create interpolation function
interp_func = interp1d(t_unique, y_fit, kind='quadratic', bounds_error=False)

# Create new time array for interpolation
t_new = np.arange(0, t_unique[-1], 0.5)
y_fit_interp = interp_func(t_new)

cfg = FilterCfg(verbose=False)
A = [component['amplitude'] for component in components]
tau = [component['tau']*1e-9 for component in components]
y_filtered = filters_wrapper(x=y_fit_interp, A=A, tau=tau, A_b=a_dc, cfg=cfg)
step = np.ones_like(y_filtered)

plt.figure(figsize=(10, 6))
plt.semilogx(t_new, y_filtered / step_factor, color='r', label='Filtered Signal with IIR', linewidth=2)
plt.semilogx(t_new, step, color='g', linestyle='-', linewidth=2, label='Ideal step')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('IIR Filter Response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0.98, 1.02])
plt.xlim([17, t_new[-1]])
plt.show()