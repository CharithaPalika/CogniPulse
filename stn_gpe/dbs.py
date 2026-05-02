import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import signal

class GenerateDBS:
  def __init__(self):
     pass
  
  def dbs_gauss_weight(self, n,c,amplitude, sigma):
      '''
      Function for gaussian distribution wts of DBS pulse

      Args:
          n(int): Grid size
          c (int): center
          amplitude (float): Amplitude
          sigma (float): spread of the gaussian pulse

      Returns:
          wt (np.ndarray): gaussian wt matrix
          (* (1/(2* np.pi * sigma**2)))
      '''
      wt = np.zeros((n,n))
      for i in range(n):
          for j in range(n):
              wt[i][j] = amplitude * np.exp(-((i-c)**2 + (j-c)**2)/(2 * sigma**2))

      return wt

  def monophasicDBS(self,amplitude: float, T: float, duty: float, sampling_freq : int, time_sec: float):
      '''
      Function to generate DBS pulse
      Args:
          amplitude (float): amplitude
          freq (float): frequency
          duty (float): duty
          sampling_freq (int): Number of samples per sec
          time_sec (float): total time in sec
      '''
      freq = 1/T
      t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
      dbs_signal = amplitude * signal.square(2 * np.pi * freq * t, duty = duty)
      #dbs_signal[dbs_signal == -1 * amplitude] = -5
      return dbs_signal

  def biphasicDBS(self, duty: float,T:float, A1:float, A2:float,  sampling_freq : int, time_sec: float, pulseinterval : float):
    """
    Generates a biphasic square pulse with variance in T (Uniform noise).

    Args:
      duty (float): duty cycle. duty must be in the interval of [0,1]
      T (float): Pulse period.
      A1 (float): Amplitude of the first phase.
      A2 (float): Amplitude of the second phase.
      sampling_freq (int): Number of samples per sec
      time_sec (float): total time in sec

    Returns:
      pulse: Biphasic square pulse waveform.
    """
    pulse_width = duty * T
    t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
    pulse1 = signal.square(2*np.pi*t/T, duty=duty/2)
    pulse2 = signal.square(2*np.pi*(t + pulseinterval * pulse_width/2)/T, duty=duty/2)
    pulse = A2/2 * pulse1 + A1/2 *  pulse2
    return pulse


  def biphasicDBS_uninoise(self,duty: float, T:float, A1:float, A2:float, low: float, high: float, sampling_freq : int, time_sec: float):
    """
    Generates a biphasic square pulse with variance in T (Uniform noise).

    Args:
      duty (float): duty cycle. duty must be in the interval of [0,1]   
      T (float): Pulse period.
      A1 (float): Amplitude of the first phase.
      A2 (float): Amplitude of the second phase.
      low (float): Noise parameter, min number
      high (float): Noise parameter, max number
      sampling_freq (int): Number of samples per sec
      time_sec (float): total time in sec
      
    Returns:
      pulse: Biphasic square pulse waveform.
    """
    t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
    uniform_rand = np.random.uniform(low=low, high= high, size = t.shape[0])
    T = T + uniform_rand
    pulse1 = signal.square(2*np.pi*t/T, duty=duty/2)
    pulse2 = signal.square(2*np.pi*(t + T/2)/T, duty=duty/2)
    pulse = A1/2 * pulse1 + A2/2 *  pulse2
    return pulse

  def biphasicDBS_normalnoise(self,duty: float, T:float, A1:float, A2:float, mean: float, 
                              std: float, sampling_freq : int, time_sec: float):
    """
    Generates a biphasic square pulse with variance in T (Uniform noise).

    Args:
      duty (float): duty cycle. duty must be in the interval of [0,1]   
      T (float): Pulse period.
      A1 (float): Amplitude of the first phase.
      A2 (float): Amplitude of the second phase.
      mean (float): Noise parameter, mean 
      std (float): Noise parameter, std deviation
      sampling_freq (int): Number of samples per sec
      time_sec (float): total time in sec

    Returns:
      pulse: Biphasic square pulse waveform.
    """
    t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
    normal_rand = np.random.normal(loc=mean, scale= std, size = t.shape[0])
    T = T + normal_rand
    pulse1 = signal.square(2*np.pi*t/T, duty=duty/2)
    pulse2 = signal.square(2*np.pi*(t + T/2)/T, duty=duty/2)
    pulse = A1/2 * pulse1 + A2/2 *  pulse2
    return pulse
  
  def biphasicDBS_noisy(self,
                        duty: float,
                        T:float,
                        amplitude:float,
                        mean: float,
                        std: float,
                        sampling_freq : int,
                        time_sec: float):
    freq = 1/T
    base_period = T
    pulse_width = duty * base_period
    base_t1 = pulse_width
    base_t2 = base_period - base_t1
    time = np.linspace(0,time_sec,int(sampling_freq*time_sec))
    
    dbs_signal = np.zeros_like(time)
    current_time = 0

    while current_time < time[-1]:
        # Add normal variability to t1 and t2
        t1 = base_t1 + np.abs(np.random.normal(mean, std))
        t2 = base_t2 + np.abs(np.random.normal(mean, std))
        
        # Ensure t1 + t2 equals the base period approximately
        period = t1 + t2
        
        # Define the start and end of each phase of the pulse
        phase1_start = current_time
        phase1_end = phase1_start + pulse_width
        phase2_start = phase1_end + t1 - pulse_width
        phase2_end = phase2_start + pulse_width

        # Create the biphasic pulse
        dbs_signal[(time >= phase1_start) & (time < phase1_end)] = amplitude
        dbs_signal[(time >= phase2_start) & (time < phase2_end)] = -amplitude
        
        # Move to the next pulse period
        current_time += period
    
    return dbs_signal
  
  def biphasicDBS_variablefreq(self,
                        pulse_width_micro : float,
                        T:float,
                        amplitude:float,
                        mean: float,
                        std: float,
                        sampling_freq : int,
                        time_sec: float):
    ''' Pulse width,std,mean in microseconds'''
    freq = 1/T
    base_period = T
    pulse_width = pulse_width_micro * 1e-6
    std = std * 1e-6
    mean = mean * 1e-6
    base_t1 = pulse_width
    base_t2 = base_period - base_t1
    time = np.linspace(0,time_sec,int(sampling_freq*time_sec))
    
    dbs_signal = np.zeros_like(time)
    current_time = 0

    while current_time < time[-1]:
        # Add normal variability to t1 and t2
        t1 = base_t1 + np.abs(np.random.normal(mean, std))
        t2 = base_t2 + np.abs(np.random.normal(mean, std))
        
        # Ensure t1 + t2 equals the base period approximately
        period = t1 + t2
        
        # Define the start and end of each phase of the pulse
        phase1_start = current_time
        phase1_end = phase1_start + pulse_width
        phase2_start = phase1_end + t1 - pulse_width
        phase2_end = phase2_start + pulse_width

        # Create the biphasic pulse
        dbs_signal[(time >= phase1_start) & (time < phase1_end)] = amplitude
        dbs_signal[(time >= phase2_start) & (time < phase2_end)] = -amplitude
        
        # Move to the next pulse period
        current_time += period
    
    return dbs_signal

  

    
