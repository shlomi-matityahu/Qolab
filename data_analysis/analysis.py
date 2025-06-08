import numpy as np
import matplotlib.pyplot as plt
normalized_amplitude = np.load("/home/shlomimatit/Projects/Qolab/data_analysis/OPX1000_LF_out_step_response_2GSaps/normalized_amplitude_ch6.npy")
print(normalized_amplitude.shape)

plt.plot(normalized_amplitude[720:])
plt.show()

