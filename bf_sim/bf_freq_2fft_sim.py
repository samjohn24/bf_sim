# +FHDR-------------------------------------------------------------------------
# FILE NAME      : bf_freq_2fft_sim.py
# AUTHOR         : Sammy Carbajal
# ------------------------------------------------------------------------------
# PURPOSE
#   Simulation of a two-dimensional FFT beamformer using PDM modulated sensors.
# -FHDR-------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time as tm
import bf_lib

def bf_freq_2fft_sim (d, dec_disable, OSR,M,c,angle_num_pts, verbose, plot_del, 
  plot_del_k, angle, ndel_max, L, r, y, fs, Do, internal):
  """
  d:                distance between sensors,
  dec_disable:      disable decimation filter,
  fsi:              input sampling rate,
  fso:              output sampling rate,
  Di:               frame length,
  M:                number of sensors,
  angle_num_pts:    number of directions to be tested,
  plot:             plot,
  verbose:          verbose,
  plot_del:         plot delay,
  plot_del_k:       angle to plot delay,
  angle:            sources arriving directions (degrees),
  f_in:             sources frequency inputs (Hertz),
  amp:              sources amplitude,
  stdv:             channel noise (standard deviation),
  mean:             channel noise (mean),
  ndel_max:         maximum delay chain length,
  L:                number of frames 
  """
  
  # ============================================================================
  #                        TWO-DIMENSIONAL FFT BEAMFORMER
  # ============================================================================

  # Number of FFT sensors
  Mi = angle_num_pts
  
  # ======================
  #   Beamformer  setup
  # ======================
  initial_time = tm.clock()

  Zld = bf_lib.bf_fft_setup(y, Mi, Do, L)
  
  # ======================
  #     One run calc
  # ======================
  
  # start loop time measure
  loop_start_time = tm.clock();
  
  # One run
  if dec_disable and not internal:
    bf_lib.bf_fft_run(Zld, np.pi, d, fs, OSR=OSR, calc_power=False)
  else:
    bf_lib.bf_fft_run(Zld, np.pi, d, fs, OSR=1, calc_power=False)

  # loop time measure 
  loop_time = tm.clock() - loop_start_time;
  
  # ===========================
  #  Direction of Arrival Calc
  # ===========================
  
  # start loop time measure
  doa_start_time = tm.clock();
  
  # DoA run
  if dec_disable and not internal:
    pbf_del, angle_bf = bf_lib.bf_fft_doa(Zld, angle_num_pts, d, fs, OSR=OSR)
  else:
    pbf_del, angle_bf = bf_lib.bf_fft_doa(Zld, angle_num_pts, d, fs, OSR=1)
  
  # loop time measure 
  doa_time = tm.clock() - doa_start_time;
  
  # ======================
  #   Final time measure
  # ======================
  
  print "Processing time:", (tm.clock()-initial_time)*1e3, "ms"
  
  return pbf_del, angle_bf
