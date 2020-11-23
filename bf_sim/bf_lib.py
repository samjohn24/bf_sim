"""
 +FHDR-------------------------------------------------------------------------
 FILE NAME      : bf_lib.py
 AUTHOR         : Sammy Carbajal
 ------------------------------------------------------------------------------
 PURPOSE
  Package for beamformer simulations.
 -FHDR-------------------------------------------------------------------------
"""
import numpy as np
import scipy as sci
import scipy.signal as sig
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# Sound speed
c = 340.

def inp_sources (fs, angle, N, f_in, amp, plot = False):
  """
    Input sources model
  
    Args:
    f_in - Sound sources frequencies in Hz 
          e.g. np.array([1e3, 3e3, 5e3)]
    angle - Arriving angles in rads e.g.  
          e.g. np.array([110., 60., 20.])*np.pi/180.
    amp  - Sound sources amplitude in units  
          e.g. np.array([1.0, 1.0, 1.0])
    N  - Number of samples
    fs - sampling rate in Hz
    plot - Plot audio sources
  
    Return:
    s   - source array
    k   - source wave vector 
  
  """
  # Wave number
  k_in_abs = 2.*np.pi*f_in/c
  
  # Number of sources
  Sn = len(angle)
  
  # source wave vector
  k = k_in_abs*np.array([[np.cos(angle),np.sin(angle)]]).reshape(2,Sn)
  
  # Source vectors
  s = np.zeros((N, Sn, 1),dtype="complex")
  
  for n in np.arange(N):
    s[n] = (amp*np.exp(1j*2*np.pi*f_in*n/fs)).reshape((Sn,1))

  
  # Plot
  if plot:
    _, axarr = plt.subplots(Sn, sharex=True)
    
    for i in np.arange(Sn):
      axarr[i].plot(np.arange(N)/fs,np.real(s[:,i,0]))
      axarr[i].grid()
      axarr[i].set_title('Source '+str(i))
    
    plt.xlabel('Time (s)')

  return s, k

def ula_setup (M, d, k, g = 1.0):
  """
  Setup Uniform Linear Array (ULA)

  Args:
  Sn - number of sound sources
  M - number of microphones
  d - distance between microphones
  k - sources wave number vector
  g - microphones directivity

  Return:
  r - sensor position vector
  a - steering vector

  """

  # Number of sources
  Sn = len(k.T)

  # sensor position vector
  r = np.zeros((M,2,1))
  
  for i in np.arange(M):
    r[i] = np.array([[d*(i-(M-1.)/2.)],[ 0.]])
  
  # steering vector
  a = np.zeros((Sn, M, 1), dtype="complex")
  
  for j in np.arange(Sn):
    for i in np.arange(M):
      a[j,i] = g*np.exp(-1j*np.matmul(r[i].transpose(),k[:,j].reshape(2,1)))
  
  a = a.reshape((Sn,M)).transpose()
    
  return r, a

def mic_array_setup (a, s, stdv, mean, plot = False):
  """
  Microphone array setup
  
  Args:
  a - steering vector
  stdv - noise std deviation
  mean - noise mean
  N    - number of samples
  M    - number of microphones
  plot - plot microphone output

  Return:
  y - microphones output real
  x - microphones output complex
  """

  # Getting parameters
  (N, _, _) = s.shape
  (M, _) = a.shape
  
  noise = np.random.normal(mean, stdv, N*M).reshape((N,M,1))
  
  # array sensor output
  x = np.zeros((N, M, 1), dtype="complex")
  y = np.zeros((N, M, 1))
  
  for n in np.arange(N):
    x[n] = np.matmul(a,s[n]) 
  
  # adding noise
  x = x + noise
  
  # array ouput
  y = np.real(x)
  
  # Plot
  if plot:
    _, axarr = plt.subplots(M, sharex=True)
    
    for i in np.arange(M):
      #axarr[i].stem(np.arange(N)/fs,y[:,i,0])
      axarr[i].stem(np.arange(N),y[:,i,0])
      axarr[i].grid()
      axarr[i].set_title('Sensor Input '+str(i))
    
    plt.xlabel('Time (s)')

  return y

def bf_fft_setup (y, Mi, D, L, plot =  False):
  """
    Setup FFT beamformer
  
    Args:
    y  - Time domain signals from microphones
    Mi - Number of microphones (Mi>=M)
    D  - Total number samples
    L  - Number of frames 
    plot - Plot FFT outputs
    dec  - Decimation rate
  
    Return
    Zl - Frequency domain array
  """

  # Leverage number of points
  Ds = D

  # Get time domain signal shape
  (N, M, _) = y.shape

  #Mi = Mi*dec

  # Zero Padding
  y_pad = np.zeros((N, Mi, 1))
  y_pad[:,:M,:] = y
  
  #  Pass to Freq. Domain
  Zl = np.zeros((L,Mi,Ds), dtype="complex")
  
  for l in np.arange(L):
    # FFT
    Zl[l,:,:] = fftpack.fft2(y_pad[Ds*l:Ds*(l+1),:,0]).transpose()

  # Plot
  if plot:
    _, axarr = plt.subplots(M, sharex=True)
    
    for i in np.arange(M):
      axarr[i].stem(np.linspace(0,fs/1e3,D),np.abs(Zl[0,i,:]))
      axarr[i].grid()
      axarr[i].set_title('Sensor Input FFT '+str(i))
    
    plt.xlabel('Freq (KHz)')

  return Zl

def bf_fft_run (Zl, angle_bf, d, fs, OSR=1, calc_power=True): 
  """
  FFT Beamformer Run
  
  Args:
  Zl       - Frequency domain data
  angle_bf - Desired direction of arrival (rads)
  d        - distance between microphones
  fs       - Sampling rate in the mic outputs
  OSR      - Oversampling rate on modulator
  calc_power - Calculate power
  
  Return
  pbf - Calculated power
  """

  #Period
  T = 1./fs

  # Data shape
  (L, Mi, Ds) = Zl.shape

  # power init
  pbf = 0

  # loop
  for l in np.arange(L):

    # Freq (in passband range)
    v = np.arange(Ds/(2*OSR), dtype="int")

    # Wavenumber sampled
    u = np.round(-np.cos(angle_bf)*Mi*d*v/(Ds*c*T)).astype(int)
    
    if calc_power:
      # accumulate power
      pbf += np.sum(np.abs(Zl[l,u,v])**2)/float(Ds/2)
  
  return pbf

def bf_fft_doa (Zl, angle_num_pts, d, fs, OSR=1): 
  """
  FFT Beamformer Run
  
  Args:
  Zl            - Frequency domain data
  angle_num_pts - Number of points in 360 degrees
  d        - distance between microphones
  fs       - Sampling rate in the mic outputs
  OSR      - Oversampling rate on modulator
  
  Return
  pbf      - Calculated power array
  angle_bf - Angle array
  """

  # power vector
  pbf = np.zeros((angle_num_pts,1))
  
  # angle vector
  angle_bf = np.linspace(0, 2*np.pi, angle_num_pts)
  
  for k in np.arange(angle_num_pts):

    # Power calculation 
    pbf[k] = bf_fft_run (Zl, angle_bf[k], d, fs, OSR)

  return pbf, angle_bf

def bf_plot_doa (pbf, angle_bf, filepath, M, N, Mi=0):
  # normalized power
  pbf_n = pbf/np.max(pbf)
  
  # Power plot
  fig = plt.figure()
  plt.plot(angle_bf*180./np.pi, pbf)
  plt.xlabel('Angle (degrees)')
  plt.title('Power')
  plt.grid()
  fig.savefig(filepath+'_power_m'+str(M)+"_mi"+str(Mi)+"_n"+str(N)+".png")
  
  # Normalized power plot (polar)
  fig = plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angle_bf, pbf_n)
  ax.grid(True)
  ax.set_title('Normalized power (polar)')
  fig.savefig(filepath+'_polar_m'+str(M)+"_mi"+str(Mi)+"_n"+str(N)+".png")
  
  plt.show()

def bf_corr_setup (y, D, L, plot = False):
  """
    Setup Correlation beamformer
  
    Args:
    y  - Time domain signals from microphones
    D  - Total number samples
    L  - Number of frames 
    plot - Plot FFT outputs
  
    Return
    Zl - Frequency domain array
  """

  # Get time domain signal shape
  (_, M, _) = y.shape

  Ylm = np.zeros((L,D,M), dtype="complex")
  
  for l in np.arange(L):
    # FFT
    Ylm[l,:,:] = fftpack.fft(y[D*l:D*(l+1)], axis=0).reshape(D,M)

  # Plot
  if plot:
    _, axarr = plt.subplots(M, sharex=True)
    
    for i in np.arange(M):
      axarr[i].stem(np.linspace(0,fs/1e3,D),np.abs(Ylm[0,i,:]))
      axarr[i].grid()
      axarr[i].set_title('Sensor Input FFT '+str(i))
  
    plt.xlabel('Freq (KHz)')

  return Ylm

def bf_corr_run (Ylm, angle_bf, r, fs, g=1.0, OSR=1, calc_power = True): 
  """
  Correlation Beamformer Run
  
  Args:
  Ylm       - Frequency domain data
  angle_bf  - Desired direction of arrival (rads)
  r         - Microphone position vector
  fs        - Sampling rate in the mic outputs
  calc_power - Calculate power
  
  Return
  pbf - Calculated power
  """
  # Get parameters
  (L, D, M) = Ylm.shape

  # Power initialize
  pbf = 0

  Dp = D/(2*OSR)
  # Freq (in passband range)
  v = np.arange(Dp, dtype="int")
  # frequency iterator
  #v = np.arange(int(D/(2*OSR)))

  # Period
  T = 1./fs
  
  # Wavenumber absolute value
  k_abs = 2.*np.pi*v.astype(float)/(float(D)*T*c)

  # ======================
  #   Steering Vector
  # ======================

  # source wave vector
  kbf = k_abs*np.array([[np.cos(angle_bf)],[np.sin(angle_bf)]])

  # reshape
  kbf = kbf.transpose().reshape(Dp,2,1)

  # steering vector
  abf = np.zeros((Dp, M, 1), dtype="complex")

  for i in np.arange(M):
    abf[:,i,:] = g*np.exp(-1j*np.matmul(r[i].T,kbf)).reshape((Dp,1))

  # abf transpose conjugated
  abf_h = abf.reshape((Dp,1,M)).conjugate()

  # weighting vector hermitian
  wh = abf_h/np.sqrt(np.sum(np.abs(abf)**2))

  # ======================
  #   Correlation Matrix
  # ======================

  # Initialize
  R = np.zeros((Dp,M,M), dtype="complex")

  for l in np.arange(L):

    # Reshape
    Yl = Ylm[l,:Dp,:]
    Yl = Yl.reshape((Dp,M,1))

    # output
    y = np.matmul(wh, Yl)

    if calc_power:

      # hermitian
      Yl_h = Yl.reshape((Dp,1,M)).conjugate()

      # Correlation calc
      R = R + np.matmul(Yl, Yl_h)
      
      # ==============
      #     Power
      # ==============

      # spectral density
      Sp = np.abs(np.matmul(np.matmul(abf_h,R),abf))
        
      # accumulate power
      pbf += np.sum(Sp)/float(D/2)

  return pbf

def bf_corr_doa (Ylm, angle_num_pts, r, fs, g=1.0, OSR=1):
  """
  Correlation Beamformer DoA
  
  Args:
  Ylm       - Frequency domain data
  angle_num_pts - Number of points in 360 degrees
  r         - Microphone position vector
  fs        - Sampling rate in the mic outputs
  g         - Mic directivity
  
  Return
  pbf      - Calculated power array
  angle_bf - Angle array
  """

  # power vector
  pbf = np.zeros((angle_num_pts,1))
  
  # angle vector
  angle_bf = np.linspace(0, 2*np.pi, angle_num_pts)
  
  # loop
  for k in np.arange(angle_num_pts):
    
    # One step run
    pbf[k] = bf_corr_run(Ylm, angle_bf[k], r, fs, g, OSR)

  return pbf, angle_bf

def sigma_delta (inp, a1=0.5, a2=0.5, vpref=6., vnref=0., plot=False, scaling=True):
  """
  Second Order Sigma Delta model

  Args:
  inp   - Modulator array input
  a1    - first coefficient
  a2    - second coefficient
  vpref - Positive voltage
  vnref - Negative voltage
  plot  - Plot first sensor output

  Return
  y     - Modulator output array
  """

  # Parameters
  (N, M, _) = inp.shape
  
  # Internal vectors
  in_sig = np.zeros((N,M))
  p_in = np.zeros((N,M))
  q_out_sig = np.zeros((N,M))
  out_sig = np.zeros((N,M))
  p1 = np.zeros((N,M))
  p2 = np.zeros((N,M))
  
  # Loop
  for n in np.arange(1,N):
    # Input signal
    in_sig[n] = inp[n].reshape(M)
     
    # Scaling to modulador range
    if scaling:
      p_in[n] = (vpref - vnref)/2.+in_sig[n]
    else:
      p_in[n] = in_sig[n]
  
    # First stage
    p1[n] = p1[n-1] + a1*(p_in[n-1] - q_out_sig[n-1])
  
    # Second stage
    p2[n] = p2[n-1] + a2*(p1[n-1] - q_out_sig[n-1])
    
    # Quantization
    for j in np.arange(M):
      if (p2[n,j] > 0):
        out_sig[n,j] = 1
        q_out_sig[n,j] = vpref
      else:
        out_sig[n,j] = -1
        q_out_sig[n,j] = vnref
  
  # Modulator output
  y = out_sig.reshape((N,M,1))
  
  # Plot
  if plot:
    plt.figure()
    plt.stem(np.arange(N),y[:,0,0])
    plt.grid()
    plt.title('First S-D output')

  return y

def decimate (inp, dec = 16, ftype='fir'):
  """
  Decimator filter

  Args:
  inp   - Filter input
  dec   - Decimation rate
  order - CIC order

  Return
  y     - Modulator output array
  """
  
  (_, M, _) = inp.shape

  for j in range(M):
    inp[::dec,j,0] = sig.decimate(inp[:,j,0],dec, ftype=ftype)

  return inp[::dec]

def interpolate (inp, factor = 16):
  """
  Interpolation filter

  Args:
  inp   - Filter input
  dec   - Decimation rate
  order - CIC order

  Return
  y     - Modulator output array
  """
  
  (a, M, b) = inp.shape
  
  outp = np.empty(shape=(a*factor,M,b))

  for j in range(M):
    outp[:,j,0] = sig.resample_poly(inp[:,j,0],factor, 1)

  return outp

def cic (inp, dec = 16, order = 5):
  """
  CIC filter

  Args:
  inp   - Filter input
  dec   - Decimation rate
  order - CIC order

  Return
  y     - Modulator output array
  """
  
  (_, M, _) = inp.shape

  h = np.ones(dec).astype(int) 
  h = np.polynomial.polynomial.polypow(h, order).astype(int)
  
  for j in range(M):
    inp[:,j,0] = sig.convolve(inp[:,j,0],h, 'same')

  return inp[::dec]

def cic_coef (dec, order = 5, Np=128):
  """
  CIC coeficient

  Args:
  dec   - Decimation rate
  order - CIC order
  Np     - FFT points

  Return
  y     - Modulator output array
  """
  # Impulse response
  h = np.ones(dec).astype(int) 
  h = np.polynomial.polynomial.polypow(h, order).astype(int)

  # Impulse response fft
  Hf = fftpack.fft(h, Np)
  
  return h , Hf

def convolve (inp, h, dec):
  """
  Convolve

  Args:
  inp   - Filter input
  h     - Filter impulse response
  dec   - Decimation rate

  Return
  y     - Modulator output array
  """
  
  (_, M, _) = inp.shape

  for j in range(M):
    inp[:,j,0] = sig.convolve(inp[:,j,0],h, 'same')

  return inp[::dec]
  
def convolve_fft (Zl, Hf, dec): 
  """
  Convolve

  Args:
  Zl    - Filter input
  Hf    - Filter impulse response
  dec   - Decimation rate

  Return
  Yf  - Convolved signal
  """

  # Data shape
  (L, Mi, D) = Zl.shape

  # Decimated
  Ddec =  D/dec

  # Reshape
  #Hf = Hf.reshape((1,1,D))

  #  Pass to Freq. Domain
  Yf = np.zeros((L,Mi,D), dtype="complex")
  Yfd = np.zeros((L,Mi,Ddec), dtype="complex")

  # loop
  for l in np.arange(L):
    for m in np.arange(Mi):
      # Convolution
      Yf[l,m,:] = Zl[l,m,:]*Hf
      
      # Frequency downsampling
      Yfd[l,m,:] = Yf[l,m,:Ddec]

  return Yfd

def bf_time_run (Ylm, angle_bf, r, fs, g=1.0, calc_power = True, verbose = False): 
  """
  Correlation Beamformer Run
  
  Args:
  Ylm       - Frequency domain data
  angle_bf  - Desired direction of arrival (rads)
  r         - Microphone position vector
  fs        - Sampling rate in the mic outputs
  calc_power - Calculate power
  
  Return
  pbf - Calculated power
  """
  # Get parameters
  (L, D, M) = Ylm.shape

  # Power initialize
  pbf = 0

  # Delay arrays
  tdel = np.zeros(M)
  ndel = np.zeros(M)

  # Period
  T = 1./fs

  # ======================
  #   Steering Vector
  # ======================

  # source wave vector
  kbf = 1./c*np.array([[np.cos(angle_bf)],[np.sin(angle_bf)]])

  # delay
  for i in np.arange(M):
    tdel[i] = -np.matmul(r[i].transpose(),rbf_u)
    ndel[i] = np.round(tdel[i]/T)

  # angle in deg
  angle_bf_deg = np.round(angle_bf[k]*180./np.pi)

  if verbose:
    print "angle_bf_deg=", angle_bf_deg
    print "ndel=", ndel

  # ===================
  #    Delay and sum
  # ===================

  z = np.zeros(D)
  Li = L/2;

  # Plot
  if plot_del and angle_bf_deg==plot_del_k:
    _, axarr = plt.subplots(M, sharex=True)

  for i in np.arange(M):
    # Delay
    y_roll = np.zeros(N)
    if np.abs(ndel[i])<=N:
      if ndel[i] == 0:
        y_roll[:] = y[:,i,0]
      elif ndel[i]>=0:
        y_roll[ndel[i]:] = y[:-ndel[i],i,0]
      else:
        y_roll[:ndel[i]] = y[-ndel[i]:,i,0]

    y_roll_f = y_roll[D*Li:D*(Li+1)]

    # Sum  
    z = z + y_roll_f

    # Plot
    if plot_del and angle_bf_deg==plot_del_k:
        axarr[i].stem(np.arange(D)/fs,y_roll_f)
        axarr[i].grid()
        axarr[i].set_title('Delayed Input '+str(i))
      
  if plot_del and angle_bf_deg==plot_del_k:
    plt.xlabel('Time (s)')
    
  # beamformer power
  pbf_del[k] = np.sum(z**2)/float(len(z))

  if verbose:
    print "pbf_del:", pbf_del[k]
