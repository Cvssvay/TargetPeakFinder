import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from typing import Any, cast

from utils.dataset import ROIDataset
from utils.roi import construct_ROI
from utils.run_utils import preprocess
import utils.peaksutils as peaks


def plot_original_signal(features, intensity):
    ''' plot original signal with shaded areas for peaks '''
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    rt_values = np.arange(len(intensity))
    smooth_intensity_values = gaussian_filter1d(intensity, sigma=1.0)
    
    ax.plot(rt_values, intensity, color='blue', label='Intensity')
    ax.plot(rt_values, smooth_intensity_values, '--', color='red', label='Filtered, sigma=1')
    
    if len(features) > 0:
        sections = [(feature[0], feature[1]) for feature in features] 
        
        # Add vertical lines and fill areas
        for section in sections:
            start, end = section
            start_rt = rt_values[start]
            end_rt = rt_values[end-1]
            section_rt = rt_values[(rt_values >= start_rt) & (rt_values <= end_rt)]
            section_intensity = intensity[(rt_values >= start_rt) & (rt_values <= end_rt)]
            ax.fill_between(section_rt, section_intensity, color=np.random.rand(3,), alpha=0.5)
    
    ax.legend(loc='upper right')
    ax.set_xlabel('RT')
    ax.set_ylabel('Intensity')
    ax.set_title('Detected Peaks with Intensity')
    
    return fig

def find_peaks_and_plots(intensity, smoothing_strength=1.0):
    ''' we want to find peaks by getting signal to noise ratios '''
    noise = peaks.estimate_noise(intensity)
    x = cast(
        np.ndarray[Any, np.dtype[np.floating]],
        gaussian_filter1d(intensity, smoothing_strength),
    ) 
    baseline = peaks.estimate_baseline(x, noise)
    find_peaks_params = {"distance": 10}
    start, apex, end = peaks.detect_peaks(x, noise, baseline, find_peaks_params)  
    n_peaks = start.size 
    features = [(s, e, a, i) for s, a, e, i in zip(start, apex, end, range(n_peaks))]
    return features

def main():
    device = torch.device('cpu')
    test_folder = '../data/test'
    test_dataset = ROIDataset(path=test_folder, device=device, interpolate=True, length=256, balanced=True)
    
    output_pdf = 'test_results.pdf'
    with PdfPages(output_pdf) as pdf:
        for cnt, dict_roi in enumerate(test_dataset.data[1]):
            roi_name = f'roi-{cnt}'

            # Debugging: Print the dict_roi to inspect its structure
            # print(f"Processing {roi_name}: {dict_roi}")
            
            # Get ROI and preprocess the intensity
            try:
                roi = construct_ROI(dict_roi)
            except TypeError as e:
                print(f"Error constructing ROI for {roi_name}: {e}")
                continue

            signal = preprocess(roi.i, device, interpolate=True, length=256)
            intensity = signal.numpy()[0][0]
            
            # Find peaks and plot
            features = find_peaks_and_plots(intensity, smoothing_strength=1.0)
            fig = plot_original_signal(features, intensity)
            
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Processed and saved {roi_name}")

if __name__ == '__main__':
    main()
