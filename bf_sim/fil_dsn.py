"""
  +FHDR------------------------------------------------------------------------
  FILE NAME      : fil_dsn.py
  DEPARTMENT     : Brazil Semiconductor Technology Center (BSTC)
  AUTHOR         : Sammy Carbajal
  AUTHORS EMAIL  : sammy.ipenza@nxp.com
  -----------------------------------------------------------------------------
  RELEASE HISTORY
  VERSION DATE        AUTHOR        DESCRIPTION
  1.1     08 Aug 2016 S. Carbajal   - Initial version.
  -----------------------------------------------------------------------------
  PURPOSE
    Library to design digital filters.
  -FHDR------------------------------------------------------------------------
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def two_comp(x, b):
    """ Convert an integer to 2's complement in integer as well
    Args:
      x -- Integer to convert
      b -- Number of bits (n)
    Returns:
      x_n -- Integer of the 2's complement format converted
    """
    return (1<<int(b))-int(-x) if x<0 else int(x)

def int2unsigned(x, b):
    """Convert integer to n-bits unsigned integer
    Args:
      x -- Integer to convert
      b -- Number of bits (n)
    Returns:
      x_n -- Integer converted to n-bits unsigned integer format
    """
    x = x.astype('int64')
    max_in = np.array(1<<b,dtype='int64')
    return (max_in +x)*(x<0)+x*(x>=0)

def int2unsigned_f(x, b):
    """Convert integer to n-bits unsigned float
    Args:
      x --  Integer to convert
      b --  Number of bits (n)
    Returns:
      x_n -- Integer converted to n-bits unsigned float format
    """
    x = x.astype('float64')
    max_in = np.array(1<<b,dtype='float64')
    return (max_in +x)*(x<0)+x*(x>=0)

def fir_lpf(fd, fc, trans, ripple_db, atten_db, nc = 90, pts_fg = 2, 
        trans_tol = 0.0, f_tol = 0.0, atten_tol_db = 0.0):
    """ Low Pass Filter Design - FIR - Remez Method
    Args:
      fd           -- Sampling frequency (Hz)
      fc           -- Cut frequency (Hz)
      trans        -- Transition band (Hz)
      ripple_db    -- Max ripple in passband (dB)
      atten_db     -- Attenuation in stopband (dB)
      nc           -- Number of coefficients
      pts_fg       -- Number of bands
      trans_tol    -- Tolerance in transition band (Hz) 
      f_tol        -- Tolerance in cut frequency (Hz), 
      atten_tol_db -- Tolerance in attenuation in stopband (dB), 
    Returns:
      lpf          -- Low pass filter coefficients
      fg           -- Bands
      ds           -- Desired gains in bands
      wt           -- Weight in bands
    """

    ripple = 2*(10**(0.5*ripple_db/20.)-1.)
    atten = 10**((atten_db-atten_tol_db)/20.)

    # Bands
    fg = (fc+f_tol)*np.linspace(0.,1.,pts_fg)
    fg = np.append(fg, [fc+trans_tol+trans, 0.5*fd])

    # Freq in bands
    ds =np.ones(int(len(fg)/2))

    ds[-1] = 0

    # Weights
    w_pass = 1./ripple
    w_stop = 1./atten

    wt = w_pass*np.ones(len(ds))
    wt[-1] = w_stop

    # Bands normalized
    fg_n = fg/fd

    # LPF design
    lpf = sig.remez(nc, fg_n, ds, wt)

    return lpf, fg_n, ds, wt

def fir_hpf(fd, fc, trans, ripple_db, atten_db, nc = 90, pts_fg = 2, 
        trans_tol = 0.0, f_tol = 0.0, atten_tol_db = 0.0):
    """ High Pass Filter Design - FIR - Remez Method
    Args:
      fd           -- Sampling frequency (Hz)
      fc           -- Cut frequency (Hz)
      trans        -- Transition band (Hz)
      ripple_db    -- Max ripple in passband (dB)
      atten_db     -- Attenuation in stopband (dB)
      nc           -- Number of coefficients
      pts_fg       -- Number of bands
      trans_tol    -- Tolerance in transition band (Hz) 
      f_tol        -- Tolerance in cut frequency (Hz), 
      atten_tol_db -- Tolerance in attenuation in stopband (dB), 
    Returns:
      hpf          -- High pass filter coefficients
      fg           -- Bands
      ds           -- Desired gains in bands
      wt           -- Weight in bands
    """
    # Transform to fc in a LPF
    fc = 0.5*fd - fc

    # LPF design
    lpf, fg, ds, wt = fir_lpf(fd, fc, trans, ripple_db, atten_db, nc, 
            pts_fg, trans_tol, f_tol, atten_tol_db)

    # Conversion to HPF
    hpf = lpf*(-1.)**np.arange(len(lpf))

    return hpf, fg, ds, wt

def cic(order, decimation, delay = 1, in_len = 2):
    """ CIC Filter Design
    Args:
      order      -- CIC order
      decimation -- CIC decimation
      delay      -- CIC delay
      in_len     -- Length of the input
    Returns:
      coeffs     -- Coefficients
      b_max      -- Max. accumulator size
    """

    K = order
    D = delay
    M = decimation

    # Coefficients
    B = np.ones(M*D)
    A = 1 

    #High Order
    B_t = B
    for _ in range(K-1):
        B_t = sig.convolve(B_t, B)

    # Last accumulator size
    b_max = np.ceil(K*np.log2(M*D)+in_len)

    coeffs = B_t

    return coeffs, b_max

def iir_hpf(fd, fmin, fmin_tol, ripple, stopband, type_dc = 'cheby1'):
    """High Pass Filter Design - IIR - First Order
    Args: 
      fd       -- Sampling frequency in Hz
      fmin     -- Minimum frequency to be into the ripple limits (Hz)
      fmin_tol -- Tolerancy to minimum frequency
      ripple   -- Ripple band (in magnitude units)
      stopband -- Maximum value of the stopband (in magnitude units)
      type_dc  -- 'cheby1' : Chebyshev type 1 filter
                  'ellip'  : Elliptical filter 
                  'cheby2' : Chebyshev type 2 filter
    Return
      b          -- Numerator coefficients
      a          -- Denominator coefficients
      shift_bits -- Number of bits to shift in implementation

    """

    # Tolerancy
    fc = fmin-fmin_tol

    #IIR design
    if type_dc == 'cheby1':
        _, a = sig.cheby1(1, ripple/2., fc/(0.5*fd), 'high')
    elif type_dc == 'ellip':
        _, a = sig.ellip(1, ripple/2, abs(stopband), fc/(0.5*fd), 'high')
    elif type_dc == 'cheby2':
        _, a = sig.cheby2(1, abs(stopband), fc/(0.5*fd), 'high')

    # NUM
    b_q = np.array([1., -1])

    # DEN
    shift_bits = int(np.ceil(np.log2(1./(1.-abs(a[1])))))
    #shift_bits = int(np.log2(1./(1.-abs(a[1]))))
    afx = 1-2**(-shift_bits)
    a_q = [1, -afx]

    return b_q, a_q, shift_bits

def iir_lpf(fd, fmin, fmin_tol, ripple, stopband, type_dc = 'cheby1'):
    """ Low Pass Filter Design - IIR - First Order
    Args: 
      fd         -- Sampling frequency in Hz
      fmin       -- Minimum frequency to be into the ripple limits (Hz)
      fmin_tol   -- Tolerancy to minimum frequency
      ripple     -- Ripple band (in magnitude units)
      stopband   -- Maximum value of the stopband (in magnitude units)
      type_dc    -- 'cheby1' : Chebyshev type 1 filter
                    'ellip'  : Elliptical filter 
                    'cheby2' : Chebyshev type 2 filter
    Return
      b          -- Numerator coefficients
      a          -- Denominator coefficients
      shift_bits -- Number of bits to shift in implementation

    """
    # High pass filter design
    b, a, shift_bits = iir_lpf(fd, fmin, fmin_tol, ripple, stopband, type_dc)

    # Convert to Low Pass
    b_q = np.array([1-a[1]])

    return b_q, a, shift_bits

def fir_linear_pred_2nd(sig, win = False, win_len  = 2**16):
    """ Linear Predictor - FIR - 2nd order
    Args:
      sig      -- signal input numpy array
      win      -- a Hann window is applied to input
      win_len  -- window length
    Returns:
      e_pred   -- prediction error
      sig_pred -- predicted signal 
      acorr    -- input autocorrelations( 0, 1, 2)
      a        -- prediction coefficients
      
    """

    # Input zero-padding
    sig     = np.append(sig, [0,0])
    # Input 1-sample delayed
    sig_l_1 = np.append([0], sig[:-2])
    # Input 2-sample delayed
    sig_l_2 = np.append([0, 0], sig[:-3])

    # Arrays declaration
    e_pred = np.zeros(len(sig))
    sig_pred = np.zeros(len(sig))
    acorr_0_reg = np.zeros(len(sig))
    acorr_1_reg = np.zeros(len(sig))
    acorr_2_reg = np.zeros(len(sig))
    a1 = np.zeros(len(sig))
    a2 = np.zeros(len(sig))

    # Window is used
    if win:
      win_len = float(win_len)
      win_frame = sig.hann(win_len) 
      
      # Window is repeated
      reps = np.ceil(len(sig)/win_len)
      win_tile = np.tile(win_frame, reps)
      win_tile = win_tile[:len(sig)]

      # Window repeated is applied
      sig = sig*win_tile

    # Autocorrelation accumulators
    acorr_0 = 0.
    acorr_1 = 0.
    acorr_2 = 0.

    for i in np.arange(2, len(sig)-1):
      
      # Reset accumulators when a window starts
      if win and not(i % win_len):
        acorr_0 = 0.
        acorr_1 = 0.
        acorr_2 = 0.

      # Autocorrelation calc
      acorr_0 += sig[i]**2.
      acorr_1 += sig[i]*sig_l_1[i]
      acorr_2 += sig[i]*sig_l_2[i]
      
      # Save autocorrelation in the time
      acorr_0_reg[i] = acorr_0
      acorr_1_reg[i] = acorr_1
      acorr_2_reg[i] = acorr_2
      
      # Calc autocorrelation, avoid division-by-0 
      if acorr_0 != 0. and acorr_1 != 0 and acorr_1 != acorr_0:
        a1[i] = (-acorr_2*acorr_1+acorr_1*acorr_0)/(acorr_1**2.-acorr_0**2.)
        a2[i] = (-acorr_1**2+acorr_2*acorr_0)/(acorr_1**2.-acorr_0**2.)
      else:
        a1[i] = a1[i-1]
        a2[i] = a2[i-1]
      
      # Prediction error calc
      e_pred[i] = a2[i]*sig_l_2[i] +a1[i]*sig_l_1[i] + sig[i]

      # Predicted signal
      sig_pred[i] =  sig[i] - e_pred[i]

      # Reset accumulators when the 2nd-derivative of all autocorrelations are 0
      if acorr_0_reg[i-2] == acorr_0_reg[i] and \
         acorr_1_reg[i-2] == acorr_1_reg[i] and \
         acorr_2_reg[i-2] == acorr_2_reg[i]:
        acorr_0 = 0.
        acorr_1 = 0.
        acorr_2 = 0.

      acorr = [acorr_0_reg, acorr_1_reg, acorr_2_reg]
      a = [a1, a2]

    return e_pred, sig_pred, acorr, a

def spectrogram(sample, rate, frame_t = 0.010):
    """ Spectrogram plot
    Args:
      sample  -- Input samples
      rate    -- Sampling frequency (Hz)
      frame_t -- Time of a frame (seconds)
    """
    period = 1./rate
    
    #Interval of spectrogram 
    time_fft = frame_t 

    # Width of spectrogram
    spec_width = int(time_fft/period)
    half_spec_wid = spec_width/2

    # Extend input
    sample_len = (len(sample)/spec_width)*spec_width
    
    n = np.arange(sample_len)
    t = n*period
    f = np.linspace(0,rate/2.,half_spec_wid)/1e3
    i = 0
    frame_2d = np.zeros((half_spec_wid,sample_len))

    # Frames calc
    while i < sample_len:
      frame = sample[i:i+spec_width]*sig.hann(spec_width)
      frame_fft = np.abs(np.fft.fft(frame))
      frame_fft_2 = frame_fft[0:half_spec_wid]
      # To dB
      frame_fft_2_log = 20*np.log10(frame_fft_2)
      frame_reshape = frame_fft_2_log.reshape((half_spec_wid,1))
      frame_2d[:,i:i+spec_width] = np.ones((1,spec_width))*frame_reshape
      i = i + spec_width

    # Center in dB scale
    frame_2d -= frame_2d.max()

    # Plot
    fig, parray = plt.subplots(2, sharex = True)
    parray[0].plot(t,sample[0:sample_len])
    parray[0].grid()
    im = parray[1].contourf(t,f,frame_2d,100)
    cfig = fig.colorbar(im, orientation= 'horizontal')

    parray[1].set_ylabel('Frequency (KHz)')
    parray[1].set_xlabel('Time (s)')

def zero_crossing (smp, fs, frmtime):
    """ Zero-crossing detection
    Args:
      smp     -- Input samples
      fs      -- Sampling frequency (Hz)
      frmtime -- Time of a frame (seconds)
    Returns:
      zcd  -- Zero-crossing count
    """

    # Length of frames
    fr_len = int(np.round(frmtime*fs))
    # Length of input
    s_len = len(smp)
    
    # Extended length
    en_len = int(np.ceil(s_len/float(fr_len)))
    s_ex_len = en_len*int(fr_len)
    
    # Extended signal - Zero padding
    signal_ex = np.zeros(s_ex_len)
    signal_ex[:s_len] = smp

    # Zero crossing array
    zcd = np.zeros(en_len)

    for i in np.arange(en_len):
      samp = signal_ex[i*fr_len:(i+1)*fr_len]
      samp_pos = samp >= 0.
      samp_neg = samp < 0.
      samp_chg = np.logical_and(samp_pos[:-1],samp_neg[1:])
      zcd[i] = len(samp_chg[samp_chg==True])

    return zcd

def energy (smp, fs, frmtime, norml1=False):
    """ Signal Energy 
    Args:
      smp     -- Input samples
      fs      -- Sampling frequency (Hz)
      frmtime -- Time of a frame (seconds)
      norm-l1 -- Use absolute value
    Returns:
      energy  -- Signal energy
    """

    # Length of frames
    fr_len = int(np.round(frmtime*fs))
    s_len = len(smp)
    
    # Extended length
    en_len = int(np.ceil(s_len/float(fr_len)))
    s_ex_len = en_len*fr_len
    
    # Extended signal - Zero padding
    signal_ex = np.zeros(s_ex_len)
    signal_ex[:s_len] = smp
    
    # Signal square
    if norml1:
      signal_sqr = np.abs(signal_ex)
    else:
      signal_sqr = signal_ex**2
    
    # Energy array
    energy = np.zeros((en_len, fr_len))
    
    # Energy calc
    energy = np.zeros((en_len, fr_len))
    for i in np.arange(fr_len):
      energy[:,i] = signal_sqr[i::fr_len]
    energy = energy.sum(axis=1)
    energy /= fr_len
    
    return energy

def plot_filter(w, h, fs = 1., title = "", norm = False, min_db = -100, 
        phase = False, ph_delay = False, gp_delay = False, ph_unwrap = False, 
        fmax_change = False, fmax = 1, hmax_override = False, hmax = 0.2, semilogx = False):
    """ Plot Filter Frequency Response
    Args:
      w         -- Angular frequency
      h         -- Magnitude
      fs        -- Sampling frequency (KHz)
      title     -- Title
      norm      -- Normalize around DC frequency
      min_db    -- Minimum decibels to show
      phase     -- Plot phase
      ph_delay  -- Plot phase delay
      gp_delay  -- Plot group delay
      ph_unwrap -- Unwrap phase (not wrap around in +-pi)
    """
    fig = plt.figure()
    if norm:
      h_db = to_db_norm(h)
    else:
      h_db = to_db(h)
    f = w/max(w)*fs/2.
    if semilogx:
      plt.semilogx(f, h_db, 'k')
    else:
      plt.plot(f, h_db, 'k')
    plt.title(title)
    plt.xlabel('Frequency(KHz)')
    plt.ylabel('dBFS')

    if not hmax_override:
      hmax = np.max(h_db)
      
    if fmax_change:
      plt.axis([0, fmax, min_db, hmax])
    else:
      plt.axis([0, 0.5*fs, min_db, hmax])
    plt.grid(True,which="both",ls="--")

    h_ph = np.angle(h)
    h_ph_unw = np.unwrap(h_ph)

    if (phase):
      if (ph_unwrap):
        h_ph_deg = h_ph_unw
      else:
        h_ph_deg = h_ph
      plt.figure()
      plt.plot(f, h_ph_deg, 'k')
      plt.title(title + " - Phase")
      plt.xlabel('Frequency(KHz)')
      plt.ylabel('Degrees (rad)')
      plt.grid()
      plt.axis([0, 0.5*fs, h_ph.min(), h_ph.max()])

    if (ph_delay):
      h_ph_del = - np.nan_to_num(h_ph_unw/(2.*np.pi*w))
      plt.figure()
      plt.plot(f, h_ph_del, 'k')
      plt.title(title + " - Phase Delay")
      plt.xlabel('Frequency(KHz)')
      plt.ylabel('Samples')
      plt.grid()
      plt.axis([0, 0.5*fs, h_ph_del.min(), h_ph_del.max()])

    if (gp_delay):
      h_gp_del = - np.nan_to_num(np.diff(h_ph_unw)/(2.*np.pi*np.diff(w)))
      plt.figure()
      plt.plot(f[:-1], h_gp_del, 'k')
      plt.title(title + " - Group Delay")
      plt.xlabel('Frequency(KHz)')
      plt.ylabel('Samples')
      plt.grid()
      plt.axis([0, 0.5*fs, h_gp_del.min(), h_gp_del.max()])

    return fig

def to_db_norm(h):
    """ Convert magnitude to decibels, normalized around DC frequency
    Args:
      h     -- Magnitude, frequency response
    Returns:
      h_db  -- Magnitude in decibels
    """
    h_db = to_db(h)
    h_db = h_db - h_db[0]
    return h_db

def to_db(h, UINF = 1000, DINF = -1000):
    """ Convert magnitude to decibels
    Args:
      h     -- Magnitude, frequency response
      UINF  -- Upward infinite value
      DINF  -- Downward infinite value
    Returns:
      h_db  -- Magnitude in decibels
    """
    h_db = 20*np.log10(abs(h)/np.nanmax(abs(h)))
    h_db[h_db == np.inf] = UINF
    h_db[h_db == -np.inf] = DINF
    h_db = np.nan_to_num(h_db)
    return h_db

def a_weighting(fs, n_pts = 100):
    """ A-Weighting filter
    Args:
      fs    -- Sampling frequency
      n_pts -- Number of points
    Returns:
      w     -- Discrete frequency (0 - pi)
      h     -- Magnitude 
    """
    w = np.linspace(0,np.pi,n_pts)
    f = w/max(w)*fs/2.

    h = 12200.**2*f**4
    h /= (f**2+20.6**2)*np.sqrt((f**2+107.7**2)*(f**2+737.9**2)*(f**2+12200**2))

    return w, h

def dyn_params(h_all, fo, fs, fo_max = False, dump = False):
    """ Dynamic parameters calculation
    Args:
      h_all  -- Frequency response (FFT result)
      fo     -- Fundamental frequency
      fs     -- Sampling frequency
      fo_max -- When is True the fundamental frequency is 
                 calculated automatically
      dump   -- Verbose
    Returns:
      snr       -- Signal to Noise Relation
      enob      -- Estimated number of bits
      thd       -- Total harmonic distortion (dB)
      noise_std -- Noise standard deviation
      Ah        -- Harmonics energy
    """
    npts = len(h_all)/2
    Np = 2.*npts
    h = abs(h_all[:npts])
    
    # Fundamental frequency
    if (fo_max):
      Nf = np.argmax(h[1:])+1.
    else:
      Nf = np.around(fo/fs*len(h_all))

    fo = Nf/len(h_all)*fs

    # Fundamental Power
    Xf = h[Nf]**2
    Af = 2.*Xf/(Np)**2
    
    # DC Power
    Xdc = np.nan_to_num(h[0]**2)
    Adc = 2.*Xdc/(Np)**2

    # Total Power
    Xt = np.nansum(h**2)
    At = 2.*Xt/(Np)**2

    # Harmonic power
    Xh = np.nansum(h[2*Nf::Nf]**2)
    Ah = 2.*Xh/(Np)**2

    # Noise Power
    Xn = Xt - Xf - Xdc - Xh
    An = 2.*Xn/(Np)**2

    # snr and enob
    snr = 10.*np.log10(Af/An)
    enob = (snr-1.76)/6.02

    # thd
    thd = 10*np.log10(Ah/Af)

    # Noise
    noise_var = An-1./12.

    # Equiv ADC bits
    equiv_adc_bits = np.log2(Af*8.)/2.

    if dump == True:
        print 'SNR:', snr
        print 'ENOB:', enob
        print 'THD(dB):', thd
        print 'THD(un):', 10.**(thd/10.)
        print 'noise_std:', np.sqrt(noise_var)
        print 'noise_var:', noise_var
        print 'Harmonics energy:', Ah
        print 'Equivalent ADC bits:', equiv_adc_bits
        print 'fo:', fo

    return snr, enob, thd, noise_var, Ah, equiv_adc_bits
