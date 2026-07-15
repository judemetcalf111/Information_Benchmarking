import pysaliency
import h5py
import numpy as np

def export_trajectory_to_hdf5(dataset_location, output_filepath, stimulus_index, subject_index):
    # 1. Load the Dataset
    # The get_mit1003 function returns both the stimuli and the fixation data
    print("Downloading/Loading MIT1003 dataset...")
    stimuli, fixations = pysaliency.get_mit1003(location=dataset_location)
    
    # 2. Isolate the Human Scanpath
    # Filter the fixations object to get the specific image (n) and subject
    mask = (fixations.n == stimulus_index) & (fixations.subjects == subject_index)
    scanpath_x = fixations.x[mask]
    scanpath_y = fixations.y[mask]
    
    # 3. Generate the Static Ground Truth Map (G)
    # This creates a standard empirical fixation density map (Gaussian blurred) 
    # to serve as your static target landscape G.
    print(f"Generating ground truth map for stimulus {stimulus_index}...")
    saliency_model = pysaliency.FixationMap(stimuli, fixations, spatial_sigma_multiplier=1.0)
    
    # Extract the map for the specific stimulus
    static_ground_truth = saliency_model.saliency_map(stimuli[stimulus_index])
    
    # Normalize the map to represent a probability distribution (summing to 1)
    static_ground_truth = static_ground_truth / np.sum(static_ground_truth)
    
    # 4. Export to HDF5
    print(f"Exporting to {output_filepath}...")
    with h5py.File(output_filepath, 'w') as h5_file:
        h5_file.create_dataset('scanpath_x', data=scanpath_x)
        h5_file.create_dataset('scanpath_y', data=scanpath_y)
        h5_file.create_dataset('static_ground_truth', data=static_ground_truth)
        
    print("Export complete.")

# Execute the export for stimulus 0 and subject 0
if __name__ == "__main__":
    export_trajectory_to_hdf5(
        dataset_location='./data', 
        output_filepath='trajectory_data.h5', 
        stimulus_index=0, 
        subject_index=0
    )
