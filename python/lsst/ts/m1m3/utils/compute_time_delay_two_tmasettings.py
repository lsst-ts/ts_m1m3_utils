import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from astropy.time import Time

from lsst.summit.utils.efdUtils import getEfdData, makeEfdClient

def displace_and_pad(array, n): #array = signal1_corrected, n=delay
    if n > 0:
        # Create a new array with the same initial value padded at the beginning
        initial_value = array[0]
        padded_array = np.full(n, initial_value)
        # Concatenate the padded array with the original array, excluding the last n elements
        result = np.concatenate((padded_array, array[:-n]))
    elif n < 0:
        ending_value = array[-1]
        padded_array = np.full(-n, ending_value)
        result = np.concatenate((array[abs(n):],padded_array))
    else:
        raise ValueError("n must be a non-zero integer")
    return result
    
def compute_time_delay(client, t1_input, t2_input, delta_t_input):
    
    t1 = Time(t1_input,scale="utc")
    t2 = Time(t2_input,scale="utc")
    delta_t = pd.to_timedelta(delta_t_input, unit="s")

    SAMPLING_FREQ = 20 # Hz
    SAMPLING_S = 1.0/SAMPLING_FREQ
    resample_rate = f'{SAMPLING_S}s' 

    df_mtmount_azi = getEfdData(
        client, "lsst.sal.MTMount.azimuth", begin=t1, end=t2+delta_t
    )

    df_mtmount_azi_upsampled = df_mtmount_azi["actualPosition"].resample(resample_rate,origin=df_mtmount_azi.index[0]).ffill() #for some reason interpolate doesn't work 

    t_start_plot = pd.to_datetime(t1.to_datetime(), utc=True)
    t_end_plot = pd.to_datetime((t1+delta_t).to_datetime(), utc=True)
    actpos_azi_t1 = df_mtmount_azi_upsampled[t_start_plot:t_end_plot]
    t_start_plot = pd.to_datetime(t2.to_datetime(), utc=True)
    t_end_plot = pd.to_datetime((t2+delta_t).to_datetime(), utc=True)
    actpos_azi_t2 = df_mtmount_azi_upsampled[t_start_plot:t_end_plot]

    signal1 = actpos_azi_t1.values
    signal2 = actpos_azi_t2.values
    if len(signal2) > len(signal1):
        signal2 = signal2[0:len(signal1)]
    elif len(signal1) > len(signal2):
        signal1 = signal1[0:len(signal2)]
    signal1_corrected = signal1 #- np.mean(signal1)
    signal2_corrected = signal2 #- np.mean(signal2)

    chi2_best = abs(np.sum((signal2_corrected - signal1_corrected) ** 2 / signal1_corrected))
    delay_keep = np.nan 
    max_delay = len(signal1_corrected)

    for delay in range(-500,500):
        if delay == 0:
            continue
        signal1_displaced = displace_and_pad(signal1_corrected, delay)
        chi2 = abs(np.sum((signal2_corrected - signal1_displaced) ** 2 / signal1_displaced))
        if chi2 < chi2_best:
            delay_keep = delay
            chi2_best = chi2

    delay_s = delay_keep / 20.
    time_delta = (t2-t1).sec
    total_delay = delay_s # + time_delta
    print(f'delay = {total_delay:.2f} s (t1 and t2 + delay should match)')
    print(f'{total_delay/60.:.2f} min')

    fig, ax = plt.subplots()
    ax.plot(signal1_corrected, label='T1')
    ax.plot(signal2_corrected, label='T2')
    signal1_displaced = displace_and_pad(signal1_corrected, delay_keep)
    ax.plot(signal1_displaced, label='displaced signal', linestyle="dashed", color='orange')
    fig.legend()
    plt.savefig('t1t2_match.png')

    return total_delay

def main():
    client = makeEfdClient()
    parser = argparse.ArgumentParser(
        description="M1M3 setting comparison (for now only time delay computation)"
    )
    parser.add_argument(
        "--t1",
        type=str,
        default="2025-09-12T09:42:30Z",
        help="Start time for first setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--t2",
        type=str,
        default="2025-09-12T09:59:50Z",
        help="Start time for second setting in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=200,
        help="Number of seconds to sample starting at t1/t2",
    )
    args = parser.parse_args()
    print(f"Using t1 = {args.t1}, t2 = {args.t2}, delta_t = {args.delta_t} s")

    time_delay = compute_time_delay(client, args.t1, args.t2, args.delta_t)
    
if __name__ == "__main__":
    main()
