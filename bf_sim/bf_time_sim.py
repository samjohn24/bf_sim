# +FHDR-------------------------------------------------------------------------
# FILE NAME      : bf_time_sim.py
# AUTHOR         : Sammy Carbajal
# ------------------------------------------------------------------------------
# PURPOSE
#   Simulation of a discrete-time beamformer using PDM modulated sensors.
# -FHDR-------------------------------------------------------------------------

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import time as tm
import bf_lib

def bf_time_sim (d, dec_disable, OSR, M, c, angle_num_pts, verbose, plot_del, 
  plot_del_k, angle, ndel_max, L, r, y, fs, Do):
  """
  d:                distance between sensors,
  dec_disable:      disable decimation filter,
  fsi:              input sampling rate,
  fso:              output sampling rate,
  OSR:              total oversampling rate
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
  L:                number of samples,
  """
  
  # ============================================================================
  #                           DISCRETE-TIME BEAMFORMER
  # ============================================================================
  
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
      ndel[i] = np.round(tdel[i]*fs)
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
          axarr[i].step(np.arange(Do)/fs,y_roll)
          axarr[i].grid()
          axarr[i].set_title('Delayed Input '+str(i))
        
    if plot_del and angle_bf_deg[k]==plot_del_k:
      plt.xlabel('Time (s)')

    if dec_disable:
      z = sig.decimate(z, OSR, ftype='fir')
      
    # beamformer power
    pbf_del[k] = np.sum(z**2)/float(len(z))
  
    if verbose:
      print "pbf_del:", pbf_del[k]
  
  # ======================
  #   Final time measure
  # ======================
  
  print "Processing time:", (tm.clock()-initial_time)*1e3, "ms"

  return pbf_del, angle_bf
  
