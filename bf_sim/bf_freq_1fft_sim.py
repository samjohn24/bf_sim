# +FHDR-------------------------------------------------------------------------
# FILE NAME      : bf_time_sim.py
# AUTHOR         : Sammy Carbajal
# ------------------------------------------------------------------------------
# PURPOSE
#   Simulation of a discrete-time beamformer using PDM modulated sensors.
# -FHDR-------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time as tm
import bf_lib

def bf_freq_1fft_sim (d, dec_disable, OSR, M,c,angle_num_pts, verbose, plot_del,
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
  #                        ONE-DIMENSIONAL FFT BEAMFORMER
  # ============================================================================
  
  # ======================
  #   Beamformer  setup
  # ======================
  initial_time = tm.clock()

  Ylm = bf_lib.bf_corr_setup(y, Do, L)
  
  # ======================
  #     One run calc
  # ======================
  
  # start loop time measure
  loop_start_time = tm.clock();
  
  # One run
  if dec_disable and not internal:
    bf_lib.bf_corr_run(Ylm, np.pi, r, fs, OSR=OSR, calc_power=False)  
  else:
    bf_lib.bf_corr_run(Ylm, np.pi, r, fs, OSR=1, calc_power=False)  

  # loop time measure 
  loop_time = tm.clock() - loop_start_time;
  
  # ===========================
  #  Direction of Arrival Calc
  # ===========================
  
  # start loop time measure
  doa_start_time = tm.clock();
  
  # DoA run
  if dec_disable and not internal:
    pbf_del, angle_bf = bf_lib.bf_corr_doa(Ylm, angle_num_pts, r, fs, OSR=OSR)
  else:
    pbf_del, angle_bf = bf_lib.bf_corr_doa(Ylm, angle_num_pts, r, fs, OSR=1)
  
  # loop time measure 
  doa_time = tm.clock() - doa_start_time;
  
  # ======================
  #   Final time measure
  # ======================
  
  print "Processing time:", (tm.clock()-initial_time)*1e3, "ms"
  
  return pbf_del, angle_bf
