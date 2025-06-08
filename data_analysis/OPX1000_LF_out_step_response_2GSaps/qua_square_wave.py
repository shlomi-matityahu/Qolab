from iqcc_cloud_client import IQCC_Cloud
from qm.qua import *
from qm import QuantumMachinesManager
import numpy as np


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                2: {
                    "type": "LF",
                    "analog_outputs": {
                        "6": {
                            "shareable": False,
                            "sampling_rate": 2000000000.0,
                            "output_mode": "amplified",
                            "offset": 0.0
                        },
                        "5": {
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0
                        },
                    }
                },
            },
        }
    },
    "elements": {
        "signal": {
            "singleInput": {"port": ["con1", 2, 6]},
        },
        "trigger": {"singleInput": {"port": ["con1", 2, 5]}},
    },
}

cloud = True

with program() as qua_program:
    # n = declare(int)
    # with for_(n, 0, n < n_rep, n+1):
    with infinite_loop_():
        set_dc_offset("trigger", "single", 0.499)
        wait(2500,"signal")
        set_dc_offset("signal","single", 1.0)
        wait(60_000, "signal")
        align("signal","trigger")
        set_dc_offset("trigger","single",0)
        set_dc_offset("signal","single", 0)
        wait(250_000,"signal")

if cloud:
    qc = IQCC_Cloud(quantum_computer_backend="qc_qolab")
    run_data = qc.execute(qua_program, config, True)# options = {"timeout":10})
else:
    qmm = QuantumMachinesManager(host='192.168.88.252',port='9510')#host=ip, port='80')  # Open quantum machine manager
    qm = qmm.open_qm(config,close_other_machines=True)  # Open quantum machine
    job = qm.execute(qua_program)  # execute QUA program

