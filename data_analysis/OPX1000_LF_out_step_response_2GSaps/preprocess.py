import matplotlib.pyplot as pl
import numpy as np
import json

# Load
filepath_base = './raw_data/ch6_amplified_no_predist_'

ts_s,amps_s,prms_s = [],[],[]
for _i in range(71):
    ts_in_sec,ys_in_V,prms = load_trace_and_setting(filepath_base+f'{_i:02.0f}')
    ts_s.append(ts_in_sec)
    amps_s.append(ys_in_V)
    prms_s.append(prms)

# Stitching
ts_stitched = []
amps_stitched = []
for _prms,_ts,_amps in zip(prms_s,ts_s,amps_s):
    amp_nrm = (_amps*1e3-_prms['offset_mV'])/(_prms['scale_mV_per_div']*5)
    idx_use = (-1<amp_nrm) * (amp_nrm<1)
    if np.sum(idx_use)!=0:
        ts_stitched += _ts[idx_use].tolist()
        amps_stitched += _amps[idx_use].tolist()
ts_stitched = np.array(ts_stitched)
amps_stitched = np.array(amps_stitched)

# Average overlapping part 
amps_single_valued = []
ts_unique = np.unique(ts_stitched)
for _t in ts_unique:
    amps_single_valued.append(np.mean(amps_stitched[ts_stitched==_t]))
amps_single_valued = np.array(amps_single_valued)

# Normalize
amp_low = np.mean(amps_single_valued[:100])
amp_high = np.mean(amps_single_valued[-100:])
amps_nrm = (amps_single_valued-amp_low)/(amp_high - amp_low)

# Plot
pl.plot(ts_unique[:1500]*1e6,amps_nrm[:1500],'o-')
pl.grid()
pl.xlabel('Trigger delay (us)')
pl.ylabel('Normalized Amplitude')
pl.show()

pl.plot(ts_unique-9.973*1e-6,amps_nrm,'o-')
pl.grid()
pl.xscale('log')
pl.xlabel('Time (s)')
pl.ylabel('Normalized Amplitude')
pl.show()

pl.plot(ts_unique-9.973*1e-6,amps_nrm,'.')
pl.ylim([0.995,1.005])
pl.grid()
pl.xscale('log')
pl.xlabel('Time (s)')
pl.ylabel('Normalized Amplitude')
pl.show()

# Save
np.save('./trigger_delay_in_seconds_ch6.npy', ts_unique)
np.save('./normalized_amplitude_ch6.npy', amps_nrm)
