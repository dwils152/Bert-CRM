import sys
from PIL import Image

# Read the list of image filenames from command line arguments
image_files = sys.argv[1:]

# Assuming the images are in a consistent size, configure the dimensions and grid
image_width, image_height = 300, 200  # Adjust based on your actual image sizes
padding = 10  # Space between images

# Number of images per row - for this example, assume you want 2 images per row
images_per_row = 2

# Calculate the number of rows needed
num_rows = (len(image_files) + images_per_row - 1) // images_per_row

# Creating a new image for the panel
panel_width = (image_width + padding) * images_per_row - padding
panel_height = (image_height + padding) * num_rows - padding
panel = Image.new('RGB', (panel_width, panel_height), 'white')

# Load and place each image
for index, filename in enumerate(image_files):
    try:
        with Image.open(filename) as img:
            # Resize image if necessary
            img = img.resize((image_width, image_height))
            # Compute the position at which to paste the image
            x_position = (index % images_per_row) * (image_width + padding)
            y_position = (index // images_per_row) * (image_height + padding)
            # Paste the image onto the panel
            panel.paste(img, (x_position, y_position))
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except IOError:
        print(f"Error opening file: {filename}")

# Save the panel of images
panel.save('panel.png')
panel.show()  # Optionally display the image
