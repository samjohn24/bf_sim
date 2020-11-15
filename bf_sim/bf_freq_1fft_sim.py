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

def bf_freq_1fft_sim (c, d, cic_osr, cic_disable, cic_order, dec_disable, fsi,
  fso, OSR, Di, M, angle_num_pts, plot, verbose, plot_del, plot_del_k, angle,
  f_in, amp, stdv, mean, ndel_max, L, save_plot_prefix):
  """
  d:                distance between sensors,
  cic_osr:          CIC filter oversampling rate,
  cic_order:        CIC filter order,
  cic_disable:      disable CIC filter,
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
  save_plot_prefix: prefix to save plot
  """
  
  # ==============================================================================
  #                             MODEL INITIALIZATION
  # ==============================================================================
  
  # Reference time
  ref_time = tm.clock()
  
  # Sound  Sources
  s, k = bf_lib.inp_sources(fsi, angle, Di, f_in, amp, plot) 
  
  # Array Setup
  r, a = bf_lib.ula_setup(M, d, k)  
  
  #   Array output
  y = bf_lib.mic_array_setup(a, s, stdv, mean, False)
  
  # Sigma delta modulator
  y_mod = bf_lib.sigma_delta(y)
  
  #  Decimation
  if dec_disable:
    y = y_mod
    fs = fsi
  else:
    #y = bf_lib.cic(y_mod, cic_osr, cic_order)
    y = bf_lib.decimate(y_mod, OSR, ftype='fir')
    fs = fso
  
  # Window length (output)
  Do = len(y[:,0,0])
  
  # Printing
  print 'Output parameters:'
  print '  {0:30}: {1:}'.format('Sampling frequency (KHz)', fso/1e3)
  print '  {0:30}: {1:}'.format('Number of samples', Do)
  
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
  bf_lib.bf_corr_run(Ylm, np.pi, r, fs, calc_power=False)  

  # loop time measure 
  loop_time = tm.clock() - loop_start_time;
  
  # ===========================
  #  Direction of Arrival Calc
  # ===========================
  
  # start loop time measure
  doa_start_time = tm.clock();
  
  # DoA run
  pbf_del, angle_bf = bf_lib.bf_corr_doa(Ylm, angle_num_pts, r, fs)
  
  # loop time measure 
  doa_time = tm.clock() - doa_start_time;
  
  # ======================
  #   Final time measure
  # ======================
  
  print "Processing time:", (tm.clock()-initial_time)*1e3, "ms"
  
  # ============================================================================
  #                                   PLOT
  # ============================================================================
  
  # normalized power
  pbf_del_n = pbf_del/np.max(pbf_del)
    
  # Power plot
  fig = plt.figure()
  plt.step(angle_bf*180./np.pi, pbf_del)
  plt.xlabel('Angle (degrees)')
  plt.title('Power')
  plt.grid()
  
  if save_plot_prefix is not None:
    filename = save_plot_prefix+'_power.png'
    fig.savefig(filename)
    print filename + ' was written.'
  
  # Normalized power plot (polar)
  fig = plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.step(angle_bf, pbf_del_n)
  ax.grid(True)
  ax.set_title('Normalized power (polar)')
  
  if save_plot_prefix is not None:
    filename = save_plot_prefix+'_polar.png'
    fig.savefig(filename)
    print filename + ' was written.'
  
  print "Total time (including plot):", (tm.clock()-initial_time)*1e3, "ms"
  plt.show()
