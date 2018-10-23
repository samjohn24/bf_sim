# +FHDR-------------------------------------------------------------------------
# FILE NAME      : bf_sim.py
# AUTHOR         : Sammy Carbajal
# ------------------------------------------------------------------------------
# PURPOSE
#   Simulation of a discrete-time beamformer using PDM modulated sensors.
# -FHDR-------------------------------------------------------------------------

import numpy as np
import argparse
from bf_time_sim import *

def main():

  # ==============================================================================
  #                           ARGUMENTS PARSING
  # ==============================================================================
  
  parser = argparse.ArgumentParser(
             description = "Discrete-time beamformer model (uniform linear array)",
             formatter_class=argparse.ArgumentDefaultsHelpFormatter
                          )
  
  # Array Maximum frequency
  parser.add_argument("--max-array-freq", type=float,
                              default=20e3,
                              help="array maximum frequency in Hertz")
  
  # Input sampling frequency
  parser.add_argument("--inp-samp-freq", type=float,
                              default=3.072e6,
                              help="input sampling frequency in Hertz")
  
  # Sound speed
  parser.add_argument("-c", "--sound-speed", type=float,
                              default=340.,
                              help="sound speed in m/s")
  
  # Distance between sensors
  parser.add_argument("-d", "--distance", type=float,
                              help="sensor distance in meters")
  
  # Window length
  parser.add_argument("-D", "--window-length", type=int,
                              default=2**13,
                              help="input window length in units")
  
  # Number of sensors
  parser.add_argument("-m", "--num-sensors", type=int,
                              default=50,
                              help="number of sensors")
  
  # Number of frames
  parser.add_argument("-L", "--num-frames", type=int,
                              default=1,
                              help="number of frames")
  
  # Number of angles (directions) tested
  parser.add_argument("-n","--num-angle-pts", type=int,
                              default=180,
                              help="number of angles (directions) tested")
  
  # Arriving sources' angles
  parser.add_argument("-a", "--angles", nargs='+', type=float, 
                              default=[110, 60, 20],
                              help="arriving sources' angles in degrees")
  
  # Arriving sources' frequencies
  parser.add_argument("-f", "--frequencies", nargs='+', type=float,
                              default=[1e3, 3e3, 5e3],
                              help="arriving sources' frequencies in Hertz")
  
  # Arriving sources' amplitudes
  parser.add_argument("-x", "--amplitude", nargs='+', type=float,
                              default=[1.0, 1.0, 1.0],
                              help="arriving sources' relative amplitudes ")
  
  # Sensor noise std dev
  parser.add_argument("--noise-stdv", type=float, default=2.0,
                              help="sensor noise standard deviation in units")
  
  # CIC disable
  parser.add_argument("--cic-disable", action='store_true', 
                              help="disable CIC filter")
  
  # CIC config
  parser.add_argument("--cic-config", nargs=2, type=int, default=[64, 5],
                              help="CIC configuration [OSR, ORDER]")
  
  # Plot
  parser.add_argument("-p", "--plot", action="store_true", default=False,
                              help="plot input source data")
  
  # Verbose
  parser.add_argument("-v", "--verbose", action="store_true", default=False,
                              help="increase output verbosity")
  
  # Plot delay
  parser.add_argument("--plot-del-angle", type=int,
                              help="plot delay angle in degrees")
  
  # Save plot
  parser.add_argument("-s","--save-plot-prefix", 
                              help="prefix to save plots")

  # Type
  parser.add_argument("--domain", type=str, default='time',
                              help="Domain (time, freq, hadam)")

  
  
  args = parser.parse_args()
  
  # ==============================================================================
  #                              SETTING PARAMETERS
  # ==============================================================================
  
  # Sound speed
  c = args.sound_speed
  
  # Distance between sensors
  if args.distance is None:
    B = args.max_array_freq
    d = c/(2*B)
  else:
    d = args.distance
    B = c/(2.*d) 
  
  # Wave number absolute value
  k_abs_max = 2.*np.pi*B/c

  # CIC disable
  cic_disable = args.cic_disable
  
  # CIC OSR
  if not cic_disable:
    cic_osr = args.cic_config[0]
  else:
    cic_osr = 1
  
  # CIC order
  cic_order = args.cic_config[1]

  # Prefix to save plot
  save_plot_prefix = args.save_plot_prefix 
  
  # Comp OSR
  comp_osr = 1
  
  # Overall OSR
  OSR = cic_osr*comp_osr
  
  # Input sampling frequency
  fsi = args.inp_samp_freq
  
  # Output sampling frequency
  fso = fsi/float(OSR)
  
  # Window length (input)
  Di = args.window_length
  
  # Number of sensors
  M = args.num_sensors
  
  # Number of frames
  L = args.num_frames
  
  # Angle number of points
  angle_num_pts = args.num_angle_pts
  
  # Plot
  plot = args.plot
  
  # Verbose
  verbose = args.verbose
  
  # Plot delay
  if args.plot_del_angle is not None:
    plot_del =  True
    plot_del_k = args.plot_del_angle
  else:
    plot_del = False
    plot_del_k = 0
  
  # Arriving angles
  angle_deg = np.array(args.angles)  
  angle = angle_deg*np.pi/180.
  
  # Input frequencies
  f_in = np.array(args.frequencies)
  
  # Amplitude
  amp = np.array(args.amplitude)
  
  # Sensor noise 
  stdv = args.noise_stdv
  mean = 0.0
  
  # Maximum delay
  tdel_max = M*d/c
  
  # Maximum delay units
  ndel_max = np.round(tdel_max*fso).astype(int)

  # Printing
  print 'Array parameters:'
  print '  {0:30}: {1:}'.format('Array type', 'ULA')
  print '  {0:30}: {1:2}'.format('Sensors distance (mm)', d*1e3)
  print '  {0:30}: {1:2}'.format('Number of sensors', M)
  print '  {0:30}: {1:2}'.format('Maximum frequency (KHz)', B/1e3)
  
  print 'Input parameters:'
  print '  {0:30}: {1:}'.format('Sampling frequency (KHz)', fsi/1e3)
  print '  {0:30}: {1:}'.format('Number of samples', Di)
  print '  {0:30}: {1:}'.format('Sources directions (degrees)', angle_deg)
  print '  {0:30}: {1:}'.format('Sources frequencies (KHz)', f_in/1e3)
  print '  {0:30}: {1:}'.format('Sources amplitude (un)', amp)
  print '  {0:30}: {1:}'.format('Channel noise (stdv) (mean=0)', stdv)
  
  if not cic_disable:
    print 'CIC filter parameters:'
    print '  {0:30}: {1:}'.format('Order', cic_order)
    print '  {0:30}: {1:}'.format('Oversampling rate', cic_osr)
  
  print 'Other parameters:'
  print '  {0:30}: {1:}'.format('Overall oversampling rate', OSR)
  print '  {0:30}: {1:}'.format('Number of frames', L)
  print '  {0:30}: {1:}'.format('Wave number (Hz*s/m)', k_abs_max)
  print '  {0:30}: {1:}'.format('Maximum delay (us)', tdel_max/1e-6)
  print '  {0:30}: {1:}'.format('Maximum delay units ', ndel_max)
  print '  {0:30}: {1:}'.format('Num. tested angles', angle_num_pts)
  print '  {0:30}: {1:}'.format('Sound speed (m/s)', c)
  
  if args.domain == 'time':
    bf_time_sim (c, d, cic_osr, cic_disable, cic_order, fsi, fso, Di, M, 
      angle_num_pts, plot, verbose, plot_del, plot_del_k, angle, f_in, 
      amp, stdv, mean, ndel_max, save_plot_prefix) 
  else:
    print 'Domain \''+ args.domain+ '\' not implemented yet.' 
 
if __name__ == '__main__':
  main() 
