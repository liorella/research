# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:11:02 2022

@author: USER
"""

# In[Init] = initialize Observe host IP/clock etc..
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
import matplotlib.image as mpimg

import observe_lib as tcp
import time
import general as gen
from TOF_calibration import *

camera_rate=40 #set Andor 40 Nuvu 80

debug = False
HOST = '172.16.2.110'
PORT = 7        # The port used by the server

cl = tcp.tcp_ip(PORT, HOST, debug)
#print("\n\rObserver Interface Structure Addresses:\n\r")
# cl.cl_addr.print_addr()
cl.ver_read()
cl.dcm_freq_set(camera_rate)
#cl.set_io("3.3V","OFF")
cl.reset_cl_interface(idelay=255)
cl.clk_test(reset_en=1)
cl.panel_leds("on")




# In[Load reference image] - if not used a file with all zeros is loaded
# want more advanced image reference capabilities? we are open to suggestions
# 1. tiff file structure?
# 2. recommended editor to make such a tiff file?
ref_img = cl.load_image_to_mem(from_file=True, dest_addr=cl.cl_addr.ref_image,
                               file_name="images\\ref_512x512_0.tif", ram_type="ddr")


# In[Load JSON config ] - load json config file
#cl.observe_config("andor\\calib_config.json")
cl.observe_config("andor\\sim_config.json")


# In[] - for simulator
# roi_512x512_100.txt proc_mode: 0- Dont wait until end of image 1-pixel-by-pixel + weights 2 - Wait for whole image

cl.config_data.tx_img = cl.load_image_to_mem(from_file=True, dest_addr=cl.cl_addr.tx_bram,
                                             file_name="images\\calib_pic_512x512.tiff", ram_type="bram")
#

# In[offset calibration]

TOF_in_ns, tresh = TOF_calibration()

# In[Camera ARM ] - arm the camera and wait for trigger (internal='int' or external='ext')



calib_offset = int(TOF_in_ns*1e-9/4e-9)

proc_time = int(15.000000e-3/4e-9)  # delay 10 ms from OPX
cl.camera_arm("ext", pix_count_reset=True, proc_time=proc_time+calib_offset,
              res_bits=cl.config_data.roi_num, rst_act_img=False, proc_timeout=0.3) # 'int' or 'ext' for internal or external trigger


# In[int trigger=] - send software trigger to test
cl.int_trigger(int_pulses=1, int_pri=65e-3)
cl.shoot_report()
cl.read_results(20, clear=False)


# In[test OPX data reception]
opx_wait_time = proc_time

opx_config = {
        'version': 1,

        'controllers': {
                'con1': {
                        'type': 'opx1',
                        'analog_outputs': { 1: {'offset': +0.0}},
                        'analog_inputs': { 1: {'offset': +0.0}},
                        'digital_outputs': {1: {}},
                        }                     

                },

        'elements': {
                'ch_1': {
                        'singleInput': {
                                'port': ('con1', 1)
                        },
                        'digitalInputs': {
                            'switch':{
                            'delay': 0,
                            'buffer': 0,
                            'port': ('con1', 1)
                            },
                        },
                        'intermediate_frequency': 0,#100e6,
                        'operations': {
                                'my_pulse_1': 'my_pulse_in1',
                                'my_pulse_2': 'my_pulse_in2',
                                'my_pulse_3': 'my_pulse_in3',
                                'my_pulse_4': 'my_pulse_in4'                                
                                
                        }
                },
                
                'ch_2': {
                        'singleInput': {
                                'port': ('con1', 1)
                        },
 
                        'intermediate_frequency': 0,#100e6,
                        'operations': {
                                'readout': 'readout_pulse',
                            
                        },
                        # 'digital_marker': 'ON',
                        'outputs': {
                            'out1': ('con1', 1)
                        },
                        'time_of_flight': 28, # nano
                        'smearing': 0,
                },

        },

        'pulses': {
                'my_pulse_in1': {
                        'operation': 'control',
                        'length': 100, # samples lentgh
                        'waveforms': {
                            'single': 'wf1'
                        },
                        'digital_marker': 'ON',
                },
                'readout_pulse': {
                        'operation': 'measurement',
                        'length': readout_len, # samples lentgh
                        'waveforms': {
                            'single': 'wf1'
                        },
                        'digital_marker': 'ON',
                },
                'my_pulse_in2': {
                        'operation': 'control',
                        'length': 1000,
                        'waveforms': {
                            'single': 'wf2'
                        }
                },
                'my_pulse_in3': {
                        'operation': 'control',
                        'length': 200,
                        'waveforms': {
                            'single': 'wf3'
                        }
                },
                'my_pulse_in4': {
                        'operation': 'control',
                        'length': 200,
                        'waveforms': {
                            'single': 'wf4'
                        }
                }                
        },

        'waveforms': {
                'wf1': {
                    'type': 'constant',
                    'sample': 0.0
                },
                'wf2': {
                    'type': 'constant',
                    'sample': 0.0
                },
                'wf3': {
                    'type': 'constant',
                    'sample': 0.498 
                },
                'wf4': {
                    'type': 'constant',
                    'sample': 0.4
                },
                'wf5': {
                    'type': 'constant',
                    'sample': 0.5
                }
                
        },
        
        'digital_waveforms': {
            'ON': {
                'samples': [(1,0)]
            }
        }
    
}


with program() as opx_calib_program :


    
    adc_st = declare_stream(adc_trace=True)
    play('my_pulse_1', 'ch_1', duration=70) # 4ns
    wait(opx_wait_time, 'ch_2') # 4ns
    measure('readout', 'ch_2', adc_st)
    
    
    with stream_processing():
        adc_st.input1().save('adc')
    
qmm = QuantumMachinesManager() 
my_qm=  qmm.open_qm(opx_config)
compiled_job = my_qm.compile(opx_calib_program)


for i in range(1):
    pending_job = my_qm.queue.add_compiled(compiled_job)
    job = pending_job.wait_for_execution()
    
    job.result_handles.wait_for_all_values()
    data = job.result_handles.get('adc').fetch_all()
    time.sleep(0.05)
    #tresh=-164
    number_of_bits=100
    result=opx_calib.show_data(0,tresh,number_of_bits,data)   

    
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]

plt.plot(data)
plt.show()


# In[shoot report]

cl.shoot_report()
cl.read_results(30, clear=False)

# In[QUA interface]


#from qua_configuration import config,number_of_ROI
wait_time=proc_time    # +4 is for the simulated data without image processing
number_of_ROI = 100
number_of_bits = number_of_ROI*2


readout_len = number_of_bits*16 #16ns per bit


config = {
        'version': 1,

        'controllers': {
                'con1': {
                        'type': 'opx1',
                        'analog_outputs': { 1: {'offset': +0.0}},
                        'analog_inputs': { 1: {'offset': +0.0}},
                        'digital_outputs': {1: {}},
                        }                     

                },

        'elements': {
                'ch_1': {
                        'singleInput': {
                                'port': ('con1', 1)
                        },
                        'digitalInputs': {
                            'switch':{
                            'delay': 0,
                            'buffer': 0,
                            'port': ('con1', 1)
                            },
                        },
                        'intermediate_frequency': 0,#100e6,
                        'operations': {
                                'my_pulse_1': 'my_pulse_in1',
                                'my_pulse_2': 'my_pulse_in2',
                                'my_pulse_3': 'my_pulse_in3',
                                'my_pulse_4': 'my_pulse_in4'                                
                                
                        }
                },
                
                'ch_2': {
                        'singleInput': {
                                'port': ('con1', 1)
                        },
 
                        'intermediate_frequency': 0,#100e6,
                        'operations': {
                                'readout': 'readout_pulse',
                            
                        },
                        # 'digital_marker': 'ON',
                        'outputs': {
                            'out1': ('con1', 1)
                        },
                        'time_of_flight': 28,
                        'smearing': 0,
                },

        },

        'pulses': {
                'my_pulse_in1': {
                        'operation': 'control',
                        'length': 100, # samples lentgh
                        'waveforms': {
                            'single': 'wf1'
                        },
                        'digital_marker': 'ON',
                },
                'readout_pulse': {
                        'operation': 'measurement',
                        'length': readout_len, # samples lentgh
                        'waveforms': {
                            'single': 'wf1'
                        },
                        'digital_marker': 'ON',
                        'integration_weights': {
                            'constant': 'constant',
                        }
                },
                'my_pulse_in2': {
                        'operation': 'control',
                        'length': 1000,
                        'waveforms': {
                            'single': 'wf2'
                        }
                },
                'my_pulse_in3': {
                        'operation': 'control',
                        'length': 200,
                        'waveforms': {
                            'single': 'wf3'
                        }
                },
                'my_pulse_in4': {
                        'operation': 'control',
                        'length': 200,
                        'waveforms': {
                            'single': 'wf4'
                        }
                }                
        },

        'waveforms': {
                'wf1': {
                    'type': 'constant',
                    'sample': 0.0
                },
                'wf2': {
                    'type': 'constant',
                    'sample': 0.0
                },
                'wf3': {
                    'type': 'constant',
                    'sample': 0.498 
                },
                'wf4': {
                    'type': 'constant',
                    'sample': 0.4
                },
                'wf5': {
                    'type': 'constant',
                    'sample': 0.5
                }
                
        },
        
        'digital_waveforms': {
            'ON': {
                'samples': [(1,0)]
            }
        },
        'integration_weights': {
            'constant': {
                'cosine': [1.0] * int(readout_len / 4),
                'sine': [0.0] * int(readout_len / 4),
                },
        }
    
}




with program() as trigger_camera_get_data :

    threshold = -0.0002
    
    
    ROI_vector = declare(int, size=number_of_ROI)
    data_vector = declare(fixed, size=number_of_bits)
    ROI_st = declare_stream()
    data_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)
    
    i = declare(int)
    j = declare(int)
    
    
    
    play('my_pulse_1', 'ch_1', duration=70) # 4ns
    wait(opx_wait_time-70, 'ch_1') # 4ns
    align()
    measure('readout', 'ch_2', adc_st, integration.sliced('constant', data_vector, 4, 'out1' ))
    
    #time of flight 
    assign(j,0)
    with for_(i,0, i<number_of_bits, i+2):
        with if_((data_vector[i]>threshold) & (data_vector[i+1]>threshold)): # 00 # signs are reversed because ADC has gain -1
            assign(ROI_vector[j], 0)
        with if_((data_vector[i]<threshold) & (data_vector[i+1]<threshold)): # 11
            assign(ROI_vector[j], 1)  
        with if_((data_vector[i]<threshold) & (data_vector[i+1]>threshold)): # 10 = limbo
            assign(ROI_vector[j], 2)
        with if_((data_vector[i]>threshold) & (data_vector[i+1]<threshold)): # 01 = Observe->OPX transfer error
            assign(ROI_vector[j], 3)
        
        save(data_vector[i], data_st)
        save(data_vector[i+1], data_st)
        save(ROI_vector[j], ROI_st)
        assign(j,j+1) 
    
        
    with stream_processing():    
       data_st.save_all('data')
       ROI_st.save_all('ROI')
       adc_st.input1().save('adc')
        #binary_st.save_all('binary_data')





qmm = QuantumMachinesManager() #host='127.16.2.137'
my_qm=  qmm.open_qm(config)
job = my_qm.execute(trigger_camera_get_data)
job.result_handles.wait_for_all_values()

data = job.result_handles.get('data').fetch_all()
ROI = job.result_handles.get('ROI').fetch_all()
raw_data = job.result_handles.get('adc').fetch_all()
#binary_data = job.result_handles.get('binary_data').fetch_all()
plt.figure()
plt.plot(data)
plt.figure()
plt.plot(ROI)
plt.figure()
plt.plot(raw_data)