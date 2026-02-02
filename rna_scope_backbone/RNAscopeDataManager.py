import h5py
import pandas as pd
import os
import numpy as np

class RNAscopeDataManager:
    def __init__(self, master_h5_path='master_data.h5', summary_csv='summary.csv', data_aq=False):
        """
        Initializes the RNAscopeDataManager.

        Parameters:
            master_h5_path (str): Path to the master HDF5 file.
            summary_csv (str): Path to the summary CSV file.
            data_aq (bool): If True, deletes existing summary CSV and master HDF5 files and initializes an empty DataFrame.
        """
        self.master_h5_path = master_h5_path
        self.summary_csv = summary_csv

        if data_aq:
            # Delete summary_csv if it exists
            if os.path.exists(self.summary_csv):
                try:
                    os.remove(self.summary_csv)
                    print(f"Deleted existing summary CSV file: {self.summary_csv}")
                except OSError as e:
                    print(f"Error deleting file '{self.summary_csv}': {e}")
            else:
                print(f"Summary CSV file '{self.summary_csv}' does not exist. No deletion needed.")

            # Delete master_h5_path if it exists
            if os.path.exists(self.master_h5_path):
                try:
                    os.remove(self.master_h5_path)
                    print(f"Deleted existing master HDF5 file: {self.master_h5_path}")
                except OSError as e:
                    print(f"Error deleting file '{self.master_h5_path}': {e}")
            else:
                print(f"Master HDF5 file '{self.master_h5_path}' does not exist. No deletion needed.")

            # Initialize an empty DataFrame
            self.summary_df = pd.DataFrame()
        else:
            # Load existing summary_csv if it exists, else initialize empty DataFrame
            if os.path.exists(self.summary_csv):
                try:
                    self.summary_df = pd.read_csv(self.summary_csv)
                    print(f"Loaded summary CSV file: {self.summary_csv}")
                except Exception as e:
                    print(f"Error reading file '{self.summary_csv}': {e}")
                    self.summary_df = pd.DataFrame()
            else:
                print(f"Summary CSV file '{self.summary_csv}' does not exist. Initializing empty DataFrame.")
                self.summary_df = pd.DataFrame()

    # Add other methods as needed for data management

    def add_file_data(self, file_id, file_path, h5_data):
        """
        Adds data for a single file to the HDF5 file and updates the summary CSV.

        Parameters:
        - file_id (str): Unique identifier for the file.
        - file_path (str): Path to the original file.
        - h5_data (dict): Dictionary containing all data to be saved.
        """
        with h5py.File(self.master_h5_path, 'a') as h5f:
            # Check if group already exists
            if file_id in h5f:
                print(f"Group '{file_id}' already exists. Overwriting the existing group.")
                del h5f[file_id]  # Delete existing group

            # Create a new group for the file
            grp = h5f.create_group(file_id)

            # Recursively create datasets and groups
            self._recursive_save(grp, h5_data)

        # Update summary DataFrame
        self._update_summary(file_id, file_path, h5_data)

    def load_file_data(self, file_id):
        """
        Loads data for a single file from the HDF5 file.

        Parameters:
        - file_id (str): Unique identifier for the file.

        Returns:
        - data (dict): Dictionary containing all loaded data.
        """
        if not os.path.exists(self.master_h5_path):
            print(f"HDF5 file not found at {self.master_h5_path}")
            return None

        with h5py.File(self.master_h5_path, 'r') as h5f:
            if file_id not in h5f:
                print(f"File ID '{file_id}' not found in HDF5 file.")
                return None

            grp = h5f[file_id]
            data = self._recursive_load(grp)

        return data

    def _recursive_save(self, h5_group, data):
        """
        Recursively saves data to the HDF5 group.

        Parameters:
        - h5_group (h5py.Group): Current HDF5 group.
        - data (dict or other): Data to save.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    # Create a subgroup
                    subgrp = h5_group.create_group(key)
                    self._recursive_save(subgrp, value)
                elif isinstance(value, list):
                    # ---- choose how to store the list ----
                    if all(isinstance(x, str) for x in value):
                        # Pure list of strings → use an HDF5 *string* dataset
                        dt = h5py.string_dtype(encoding="utf-8")
                        h5_group.create_dataset(key, data=value, dtype=dt)

                    else:
                        # Try to convert to a regular NumPy array (numbers, bools, etc.)
                        try:
                            arr = np.asarray(value)
                            # Reject ragged/mixed lists that turn into dtype=object
                            if arr.dtype == object:
                                raise TypeError
                            h5_group.create_dataset(key, data=arr)
                        except Exception:
                            # Fallback: store a UTF-8 string representation so nothing is lost
                            h5_group.create_dataset(
                                key,
                                data=np.bytes_(str(value)),
                                dtype=h5py.string_dtype("utf-8")
                            )
                elif isinstance(value, (np.ndarray, np.generic)):
                    # Directly save NumPy arrays
                    h5_group.create_dataset(key, data=value)
                elif isinstance(value, str):
                    # Handle string data
                    dt = h5py.string_dtype(encoding='utf-8')
                    h5_group.create_dataset(key, data=value, dtype=dt)
                elif value is None:
                    # Save as empty dataset
                    print(f"Value for key '{key}' is None. Saving as empty dataset.")
                    h5_group.create_dataset(key, data=np.array([]))
                else:
                    # Attempt to save other data types
                    try:
                        h5_group.create_dataset(key, data=value)
                    except TypeError:
                        # Convert to string if not compatible
                        h5_group.create_dataset(key, data=np.bytes_(str(value)))
        elif isinstance(data, list):
            # Save lists as datasets
            try:
                array_data = np.array(data)
                h5_group.create_dataset('list_data', data=array_data)
            except:
                print("Failed to convert list to NumPy array.")
                h5_group.create_dataset('list_data', data=np.bytes_(str(data)))
        else:
            # Save other data types as datasets
            try:
                h5_group.create_dataset('data', data=data)
            except TypeError:
                # Convert to string if not compatible
                h5_group.create_dataset('data', data=np.bytes_(str(data)))

    def _recursive_load(self, h5_group):
        """
        Recursively loads data from the HDF5 group.

        Parameters:
        - h5_group (h5py.Group): Current HDF5 group.

        Returns:
        - data (dict or other): Loaded data.
        """
        data = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):
                # Recursively load subgroup
                data[key] = self._recursive_load(item)
            elif isinstance(item, h5py.Dataset):
                # Load dataset
                data[key] = item[()]
                # Decode byte strings if necessary
                if item.dtype.kind == 'S':  # Byte string
                    if isinstance(data[key], bytes):
                        data[key] = data[key].decode('utf-8')
                    elif isinstance(data[key], np.ndarray) and data[key].dtype.kind == 'S':
                        data[key] = np.char.decode(data[key], encoding='utf-8')
        # Load attributes (metadata)
        for attr_key, attr_value in h5_group.attrs.items():
            if isinstance(attr_value, bytes):
                data[attr_key] = attr_value.decode('utf-8')
            else:
                data[attr_key] = attr_value
        return data

    def _update_summary(self, file_id, file_path, h5_data):
        """
        Updates the summary DataFrame with metadata from h5_data.

        Parameters:
        - file_id (str): Unique identifier for the file.
        - file_path (str): Path to the original file.
        - h5_data (dict): Dictionary containing all data to be saved.
        """
        # Extract metadata
        metadata = h5_data.get('metadata', {})
        summary_entry = {
            'file_id': file_id,
            'file_path': file_path,
            # Add other metadata fields as needed
        }

        # Flatten metadata if necessary
        for key, value in metadata.items():
            summary_entry[key] = value

        # Convert cluster intensities to list if they are numpy arrays
        if 'cluster_intensities_mouse' in h5_data:
            summary_entry['cluster_intensities_mouse'] = h5_data['cluster_intensities_mouse'].tolist()
        if 'cluster_intensities_first_exon' in h5_data:
            summary_entry['cluster_intensities_first_exon'] = h5_data['cluster_intensities_first_exon'].tolist()

        # Create a DataFrame for the new entry
        summary_entry_df = pd.DataFrame([summary_entry])

        # Concatenate the new entry to the existing DataFrame
        self.summary_df = pd.concat([self.summary_df, summary_entry_df], ignore_index=True)

        # Save the updated DataFrame to CSV
        self.summary_df.to_csv(self.summary_csv, index=False)
