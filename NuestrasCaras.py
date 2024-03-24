import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import pandas as pd

# Define the folder path
# Get the current working directory
current_directory = os.getcwd()
folder_name = "fotos"
folder_path = os.path.join(os.path.dirname(os.path.abspath(__name__)), folder_name)
print(folder_path)

# Get a list of file names in the folder
file_names = os.listdir(folder_path)

# Initialize an empty list to store the images
images = []

# Iterate over the file names
for file_name in file_names:
    # Construct the file path
    file_path = os.path.join(folder_path, file_name)
    
    # Load the image using PIL
    image = Image.open(file_path)
    
    # Convert the image to greyscale
    image = image.convert('L')
    
    # Resize the image to 30x30 pixels
    image = image.resize((30, 30))
    
    # Append the image to the list
    images.append(image)
    

# Display the first image
#images[0].show()

# So far we have a list of images

# Initialize an empty list to store the greyscale values for all images
all_greyscale_values = []

# Iterate over the images
for image in images:
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Extract the greyscale values for each pixel
    greyscale_values = image_array.flatten()
    
    # Append the greyscale values to the list
    all_greyscale_values.append(greyscale_values)

# Create a DataFrame to store all the greyscale values
df = pd.DataFrame(all_greyscale_values)

# Save the DataFrame to a CSV file
df.to_csv('greyscale_values.csv', index=False) # Este archivo tiene todos los valores de los pixeles de las fotos