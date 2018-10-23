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

def bf_time_sim (c, d, cic_osr, cic_disable, cic_order, fsi, fso, Di, M, 
  angle_num_pts, plot, verbose, plot_del, plot_del_k, angle, f_in, amp, stdv, 
  mean, ndel_max, save_plot_prefix):
  """
  d:                distance between sensors,
  cic_osr:          CIC filter oversampling rate,
  cic_order:        CIC filter order,
  cic_disable:      disable CIC filter,
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
  if cic_disable:
    y = y_mod
  else:
    y = bf_lib.cic(y_mod, cic_osr, cic_order)
  
  # Window length (output)
  Do = len(y[:,0,0])
  
  # Printing
  print 'Output parameters:'
  print '  {0:30}: {1:}'.format('Sampling frequency (KHz)', fso/1e3)
  print '  {0:30}: {1:}'.format('Number of samples', Do)
  
  # ==============================================================================
  #                           DISCRETE-TIME BEAMFORMER
  # ==============================================================================
  
  # =========================
  #  Conventional Beamformer
  # =========================
  
  # power vector
  pbf_del = np.zeros((angle_num_pts,1))
  
  # angle vector
  angle_bf = np.linspace(0, 2*np.pi, angle_num_pts)
  
  # angle in deg
  angle_bf_deg = np.round(angle_bf*180./np.pi)
  
  # delay vector
  tdel = np.zeros((M, 1))
  ndel = np.zeros(M, dtype=int)
  ndel_norm = np.zeros(M, dtype=int)
  
  # ======================
  #  Start time measure
  # ======================
  initial_time = tm.clock()
  
  # Loop
  for k in np.arange(angle_num_pts):
    # =============
    #    Delay 
    # =============
  
    # source wave vector
    rbf_u = 1./c*np.array([[np.cos(angle_bf[k]),np.sin(angle_bf[k])]]).transpose()
    
    # delay
    for i in np.arange(M):
      tdel[i] = -np.matmul(r[i].transpose(),rbf_u)
      ndel[i] = np.round(tdel[i]*fso)
      ndel_norm[i] = ndel[i] + ndel_max/2
  
    if verbose:
      print "angle_bf_deg=", angle_bf_deg[k]
      print "ndel=", ndel
      print "ndel_norm=", ndel_norm
  
    # ===================
    #    Delay and sum
    # ===================
  
    z = np.zeros(Do)
  
    # Plot
    if plot_del and angle_bf_deg[k]==plot_del_k:
      _, axarr = plt.subplots(M, sharex=True)
  
    for i in np.arange(M):
      # Delay
      y_roll = np.zeros(Do)
      if ndel_norm[i] == 0:
        y_roll[:] = y[:,i,0]
      elif ndel_norm[i]>0:
        y_roll[ndel_norm[i]:] = y[:-ndel_norm[i],i,0]
  
      # Sum  
      z = z + y_roll
  
      # Plot
      if plot_del and angle_bf_deg[k]==plot_del_k:
          axarr[i].step(np.arange(Do)/fso,y_roll)
          axarr[i].grid()
          axarr[i].set_title('Delayed Input '+str(i))
        
    if plot_del and angle_bf_deg[k]==plot_del_k:
      plt.xlabel('Time (s)')
      
    # beamformer power
    pbf_del[k] = np.sum(z**2)/float(len(z))
  
    if verbose:
      print "pbf_del:", pbf_del[k]
  
  # ======================
  #   Final time measure
  # ======================
  
  print "Processing time:", (tm.clock()-initial_time)*1e3, "ms"
  
  # ==============================================================================
  #                                   PLOT
  # ==============================================================================
  
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
