# bf_sim

```
usage: bf_sim [-h] [--max-array-freq MAX_ARRAY_FREQ] [--inp-samp-freq INP_SAMP_FREQ] [--dec-disable]
              [--out-samp-freq OUT_SAMP_FREQ] [-c SOUND_SPEED] [-d DISTANCE] [-D WINDOW_LENGTH] [-m NUM_SENSORS]
              [-L NUM_FRAMES] [-n NUM_ANGLE_PTS] [-a ANGLES [ANGLES ...]] [-f FREQUENCIES [FREQUENCIES ...]]
              [-x AMPLITUDE [AMPLITUDE ...]] [--noise-stdv NOISE_STDV] [--cic-disable]
              [--cic-config CIC_CONFIG CIC_CONFIG] [-p] [-v] [--plot-del-angle PLOT_DEL_ANGLE] [-s SAVE_PLOT_PREFIX]
              [--disable-plot-title] [--pdf] [--no-show] [--internal] [--intp-factor INTP_FACTOR] [--method METHOD]

Discrete-time beamformer (uniform linear array) simulator

optional arguments:
  -h, --help            show this help message and exit
  --max-array-freq MAX_ARRAY_FREQ
                        array maximum frequency in Hertz (default: 20000.0)
  --inp-samp-freq INP_SAMP_FREQ
                        input sampling frequency in Hertz (default: 3072000.0)
  --dec-disable         disable decimation filter (default: False)
  --out-samp-freq OUT_SAMP_FREQ
                        output sampling frequency in Hertz (default: 48000.0)
  -c SOUND_SPEED, --sound-speed SOUND_SPEED
                        sound speed in m/s (default: 340.0)
  -d DISTANCE, --distance DISTANCE
                        sensor distance in meters (default: None)
  -D WINDOW_LENGTH, --window-length WINDOW_LENGTH
                        input window length in ms (default: 4)
  -m NUM_SENSORS, --num-sensors NUM_SENSORS
                        number of sensors (default: 50)
  -L NUM_FRAMES, --num-frames NUM_FRAMES
                        number of frames (default: 1)
  -n NUM_ANGLE_PTS, --num-angle-pts NUM_ANGLE_PTS
                        number of angles (directions) tested (default: 180)
  -a ANGLES [ANGLES ...], --angles ANGLES [ANGLES ...]
                        arriving sources' angles in degrees (default: [110, 60, 20])
  -f FREQUENCIES [FREQUENCIES ...], --frequencies FREQUENCIES [FREQUENCIES ...]
                        arriving sources' frequencies in Hertz (default: [1000.0, 3000.0, 5000.0])
  -x AMPLITUDE [AMPLITUDE ...], --amplitude AMPLITUDE [AMPLITUDE ...]
                        arriving sources' relative amplitudes (default: [1.0, 1.0, 1.0])
  --noise-stdv NOISE_STDV
                        sensor noise standard deviation in units (default: 2.0)
  --cic-disable         disable CIC filter (default: False)
  --cic-config CIC_CONFIG CIC_CONFIG
                        CIC configuration [CICOSR, ORDER] (default: [64, 5])
  -p, --plot            plot input source data (default: False)
  -v, --verbose         increase output verbosity (default: False)
  --plot-del-angle PLOT_DEL_ANGLE
                        plot delay angle in degrees (default: None)
  -s SAVE_PLOT_PREFIX, --save-plot-prefix SAVE_PLOT_PREFIX
                        prefix to save plots (default: None)
  --disable-plot-title  disable plot title (default: False)
  --pdf                 save plots in PDF format (default: False)
  --no-show             no show plots interactively (default: False)
  --internal            show internal power (before decimation) (default: False)
  --intp-factor INTP_FACTOR
                        interpolation factor (default: None)
  --method METHOD       Domain (time, freq_1fft, freq_2fft, hadam) (default: time)
```

## Demo

In 'tests' folder run './bf_sim'

