import torch
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator
from utils.run_utils import preprocess, get_borders, Feature
import numpy as np
from utils.dataset import ROIDataset
from utils.roi import construct_ROI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

class EvaluationModel:
    def __init__(self, test_folder_path, peak_minimum_points):
        self.test_folder = test_folder_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path2classifier_weights = '../data/tmp_weights/Classifier'
        self.path2segmentator_weights = '../data/tmp_weights/Segmentator'
        self.peak_minimum_points = peak_minimum_points
        self.load_models()

    def load_models(self):
        ''' load classifier and segmentation models'''
        print('Model is loading')
        classifier = Classifier().to(self.device)
        classifier.load_state_dict(torch.load(self.path2classifier_weights, map_location=self.device))
        classifier.eval()
        segmentator = Segmentator().to(self.device)
        segmentator.load_state_dict(torch.load(self.path2segmentator_weights, map_location=self.device))
        segmentator.eval()
        self.classifier = classifier
        self.segmentator = segmentator

    def get_features(self, dict_roi, metName='tempmet'):
        roi = construct_ROI(dict_roi)
        
        # Preprocess the intensity
        signal = preprocess(roi.i, self.device, interpolate=True, length=256)
        
        # Interpolate rt to match the length of 256
        interp_func = interp1d(np.arange(len(roi.rt)), roi.rt, kind='linear')
        interpolated_rt = interp_func(np.linspace(0, len(roi.rt) - 1, 256))

        # Predict by classifiers 
        classifier_output, _ = self.classifier(signal)
        # Predict by segmentator 
        _, segmentator_output = self.segmentator(signal)

        classifier_output = classifier_output.data.cpu().numpy()
        segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()

        # Get label
        label = np.argmax(classifier_output)
        # Get borders
        features = []
        if label == 1:
            borders = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                  peak_minimum_points=self.peak_minimum_points,
                                  interpolation_factor=len(signal[0, 0]) / len(roi.i))
            for border in borders:
                scan_frequency = (roi.scan[1] - roi.scan[0]) / (roi.rt[1] - roi.rt[0])
                rtmin = roi.rt[0] + border[0] / scan_frequency
                rtmax = roi.rt[0] + border[1] / scan_frequency
                feature = Feature([metName], [roi], [border], [0], [np.sum(roi.i[border[0]:border[1]])],
                                  roi.mzmean, rtmin, rtmax, 0, 0)
                features.append(feature)
        return features, interpolated_rt, signal.cpu().numpy()[0, 0, :]

    def plot_feature(self, roi, features, title, interpolated_rt, interpolated_intensity):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot the original signal
        ax1.plot(interpolated_rt, interpolated_intensity)
        ax1.set_title(f"{title} - Original Signal")
        ax1.set_xlabel('Retention Time (rt)')
        ax1.set_ylabel('Intensity')
        
        # Plot the features
        ax2.set_title(title)
        if features:
            for feature in features:
                feature.plot(ax2, shifted=False, show_legend=True)
        else:
            ax2.text(0.5, 0.5, 'No Features Detected', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return fig

def main():
    
    device = torch.device('cpu')
    test_folder = '../data/test'
    peak_minimum_points = 3
    test_dataset = ROIDataset(path=test_folder, device=device, interpolate=True, length=256, balanced=True)
    evalmodel = EvaluationModel(test_folder, peak_minimum_points)
    
    with PdfPages('features.pdf') as pdf:
        cnt = 0
        for dict_roi in test_dataset.data[1]:
            import pdb;pdb.set_trace()
            roi_name = f'roi-{cnt}'
            cnt += 1
            features, interpolated_rt, interpolated_intensity = evalmodel.get_features(dict_roi, roi_name)
            title = f'{roi_name}'
            fig = evalmodel.plot_feature(dict_roi, features, title, interpolated_rt, interpolated_intensity)
            pdf.savefig(fig)
            plt.close(fig)

if __name__ == '__main__':
    main()
