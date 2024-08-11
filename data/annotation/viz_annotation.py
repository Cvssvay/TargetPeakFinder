import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
def plot_intensity_with_vertical_line(intensity, sections, pdf_pages):
    """
    Plot intensity array and add vertical lines at specified x-axis values.

    Parameters:
    - intensity: list or array-like, the intensity values to plot.
    - sections: list of tuples, each containing the start and end index of a section.
    - pdf_pages: PdfPages object to save the plots.
    """
    # X-axis values
    x_values = range(len(intensity))

    # Plot intensity
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, intensity, label='Intensity', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Intensity')
    plt.title('Intensity Plot')
    
    # Add vertical lines at the specified x-axis values for each section
    for section in sections:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate random hex color
        plt.axvline(x=section[0], color=color, linestyle='--', label='Vertical Line')
        plt.axvline(x=section[1], color=color, linestyle='--')


    plt.legend()
    plt.grid(True)
    
    # Save the plot to the PDF page
    pdf_pages.savefig()
    plt.close()

def list_and_read_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    with PdfPages('plots.pdf') as pdf:
        for filename in json_files:
            filepath = os.path.join(directory, filename)
            print("Reading file:", filepath)
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                    plot_intensity_with_vertical_line(data['intensity'], data['borders'], pdf)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON in file", filename, ":", str(e))

# Example usage:
directory_path = '/Users/vpandey/projects/gitlabs/peakDetection/bory/annotation/OriginalData_001'
list_and_read_json_files(directory_path)
