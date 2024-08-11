import os
import pandas as pd
import numpy as np
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
# Step 2.1: Read the sample metadata file
metadata = pd.read_csv('sample_metadata.csv')

# Step 2.2: Initialize an empty DataFrame to store the results
columns = ['sample_id', 'sample_name', 'scan', 'RT', 'intensity', 'polarity']
results_df = pd.DataFrame(columns=columns)

# Directory where the mzML files are stored
mzml_directory = '/Users/vpandey/projects/gitlabs/metavyom2/metavyom21/sarvaomics/static/uploadData'  # Update this path to your mzML files directory

def read_targeted_excel():
    ''' read target excel file'''
    df=pd.read_excel('/Users/vpandey/projects/gitlabs/peakDetection/bory/data/HILIC_MasterMethod_Kanarek_20201028.xlsx',sheet_name='bory')
    return df

def get_intensity_based_mz_and_ppm(tdf, df, ppm, rt_inv, rt_mean_interval, min_intensity, key_column='Compound Name'):
    '''
    For each mz value, extract intensities.
    '''
    # Convert 'PeakPolarity' to flag
    tdf['flag'] = tdf['PeakPolarity'].map({'Positive': 1, 'Negative': 2})

    # Calculate mztol
    tdf['mztol'] = tdf['m/z'] * ppm * 1e-6
    tdf['Retention Time'] = tdf['Retention Time'] * 60  # in seconds
    # tdf['rttol'] = int(rt_inv / rt_mean_interval)
    tdf['rttol'] = rt_inv
    # Define a function to filter df based on conditions
    def filter_df(row):
        mz = row['m/z']
        mztol = row['mztol']
        flag = row['flag']
        rttol = row['rttol']
        rt = row['Retention Time']
        
        # Filter df based on conditions
        filtered_df = df[(df['mz'] < mz + mztol) & 
                         (df['mz'] > abs(mz - mztol)) & 
                         (df['RT'] < rt + rttol) & 
                         (df['RT'] > abs(rt - rttol)) & 
                         (df['polarity'] == flag) & 
                         (df['inty'] > min_intensity)]
        
        # If filtered_df is not empty, find max_intensity, max_intensity_RT, and max_intensity_mz
        if not filtered_df.empty:
            max_intensity_row = filtered_df.loc[filtered_df['inty'].idxmax()]
            max_intensity = max_intensity_row['inty']
            max_intensity_rt = max_intensity_row['RT']
            max_intensity_mz = max_intensity_row['mz']
        else:
            # Set max_intensity, max_intensity_RT, and max_intensity_mz to None if filtered_df is empty
            max_intensity = None
            max_intensity_rt = None
            max_intensity_mz = None
        
        return (filtered_df, max_intensity, max_intensity_rt, max_intensity_mz)

    # Apply the filter_df function to each row of tdf and store the results in a dictionary
    targeted_int_dict = tdf.apply(lambda row: (row[key_column], *filter_df(row)), axis=1)
    
    # Remove empty filtered DataFrames from the resulting dictionary
    targeted_int_dict = {key: (filtered_df, max_intensity, max_intensity_rt, max_intensity_mz) 
                         for key, filtered_df, max_intensity, max_intensity_rt, max_intensity_mz in targeted_int_dict 
                         if not filtered_df.empty}

    return targeted_int_dict



def get_range_targeted_excel(tdf,df,rt, polarity,rt_mean_interval,ppm=10,rt_interval=40,min_intensity=10000,polarity_flag=1):
    '''
    read range targeted
    tdf is targted metabolimcs data 
    df is the dataframe for RT,mz,intensity,polarity
    rt is the list for retention times for each run 
    polarity is the list for polarity for each run
    rt_mean_interval is the mean over all intervals
    ppm is parts per million
    rt_interval is rtinterval in second 
    min_intensity minimum intensity for intensity
    polarity_flag is 1 for positive and 2 for negative
    '''
    ## get list of intensities for targeted compounds 
    targeted_int_df=get_intensity_based_mz_and_ppm(tdf,df,ppm,rt_interval,rt_mean_interval,min_intensity)
    ### apply get range 
    h_rt = []
    h_ms = []
    h_intensity = []
    choose_spec = []
    metabolites=[]
    for k,v in targeted_int_df.items():
        metabolites.append(k)
        tdf= v[0][['RT','mz','inty']]         
        h_rt.append(v[2])
        choose_spec.append(tdf.values)  
        h_intensity.append(v[1])
        h_ms.append(v[3])
      
    spec_rt = list(zip(choose_spec,h_rt,h_ms,h_intensity))
    return  spec_rt,metabolites
       


def readms(file_path):
    """
    Read mzXML, mzML and mzData files.
    Arguments:
        file_path: string
            path to the dataset locally
    Returns:
        Tuple of Numpy arrays: (m/z, intensity, retention time, mean interval of retention time).
    
    
    """
    ms_format = os.path.splitext(file_path)[1]
    ms_format = ms_format.lower()
    exp = MSExperiment()
    if ms_format == '.mzxml':
        file = MzXMLFile()
    elif ms_format == '.mzml':
        file = MzMLFile()
    elif ms_format == '.mzdata':
        file = MzDataFile()
    file.load(r'%s' % file_path, exp)
    mz_list = []
    intensity_list = []
    rt_list = []
    polarity_list=[]
    rtAll = []
    polarityAll=[]
    # Iterate over each spectrum in exp and append data to lists
    for spec in exp:
        if spec.getMSLevel() == 1:
            mz, intensity = spec.get_peaks()
            rtAll.append(spec.getRT())
            rt = np.full([mz.shape[0]], spec.getRT(), float)
            polarity = np.full([mz.shape[0]],spec.getInstrumentSettings().getPolarity() , int)
            mz_list.append(mz)
            polarityAll.append(spec.getInstrumentSettings().getPolarity())
            intensity_list.append(intensity)
            rt_list.append(rt)
            polarity_list.append(polarity)

    # Concatenate lists to create arrays
    mz_array = np.concatenate(mz_list)
    intensity_array = np.concatenate(intensity_list)
    rt_array = np.concatenate(rt_list)
    polarity_array = np.concatenate(polarity_list)
    # ["RT", "mz", "inty"]
    # Create DataFrame from arrays
    df = pd.DataFrame({
        'mz': mz_array,
        'inty': intensity_array,
        'RT': rt_array,
        'polarity':polarity_array
    })

    df=df[df['inty']>0]
    rt2 = np.array(rtAll)
    if rt2.shape[0] > 1:
        rt_mean_interval = np.mean(np.diff(rt2))
    else:
        rt_mean_interval = 0.0 
    return df, rtAll, polarityAll,rt_mean_interval




# Step 2.3: Function to read mzML file and extract data
def process_mzml(file_path, sample_id, sample_name):

    dfm, rtAll, polarityAll,rt_mean_interval=readms(file_path)
    dft=read_targeted_excel()
    choose_spec,metabolites=get_range_targeted_excel(dft,dfm,rtAll,polarityAll,rt_mean_interval,ppm=10,rt_interval=40,min_intensity=10000,polarity_flag=1)
    # "RT","mz","intensity"
    
   

for _, row in metadata.iterrows():
    file_path = os.path.join(mzml_directory, row['sample_name'])
    process_mzml(file_path, row['sample_id'], row['sample_name'])

    


