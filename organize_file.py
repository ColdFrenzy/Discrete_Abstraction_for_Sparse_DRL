import os
import shutil

# Define the root directory where all the 7x7_# directories are located
root_dir = 'sa_sac_uav_tensorboard_gamma98' 

# Define the new directory where we want to move the files
new_dir = os.path.join(root_dir, '7x7')

# Create the new 7x7 directory if it doesn't exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Loop through each item in the root directory
for item in os.listdir(root_dir):
    # Check if the item is a directory starting with '7x7_'
    if item.startswith('7x7_'):
        # Get the number part from '7x7_#'
        dir_number = item.split('_')[1]
        sac_1_path = os.path.join(root_dir, item, 'SAC_1')
        
        # Check if SAC_1 exists
        if os.path.exists(sac_1_path):
            # Path to the file inside SAC_1
            sac_1_files = os.listdir(sac_1_path)
            
            if sac_1_files:
                # Assume there's only one file inside SAC_1
                file_name = sac_1_files[0]  # Get the first file in SAC_1
                file_path = os.path.join(sac_1_path, file_name)
                
                # Create a new subdirectory for the number inside the new '7x7' directory
                new_subdir = os.path.join(new_dir, dir_number)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)
                
                # Move the file to the new subdirectory
                shutil.move(file_path, os.path.join(new_subdir, file_name))
                print(f"Moved {file_path} to {new_subdir}/{file_name}")
            else:
                print(f"No files found in {sac_1_path}")
        else:
            print(f"SAC_1 directory does not exist in {item}")

print("Process completed.")