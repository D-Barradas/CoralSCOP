import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import io
import plotly.express as px

# Import necessary libraries for deep learning models (e.g., TensorFlow, PyTorch)

with open("load_functions.py") as f:
    exec(f.read())


def is_cuda_available():
    """Checks if CUDA is available and can be used by PyTorch.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """

    return torch.cuda.is_available()

# Function to load a model based on selection
def load_model_and_segment(image, model_option):

    if model_option == 'Model_H':
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    elif model_option == 'Model_L':
        sam_checkpoint = "models/sam_vit_l_0b3195.pth"  # this is 31% of the memory of a V100
        model_type = "vit_l"
    else:  # Model_C
        sam_checkpoint = "models/sam_vit_b_01ec64.pth"  # this is even less
        model_type = "vit_b"

    #device = torch.device("cuda:0")
    if is_cuda_available():
        st.markdown("CUDA is available!")
        device = torch.device("cuda")
    else:
        st.markdown("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    return masks



# Function to create and store the plot in session state
def create_plot(image, masks):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('on')

    # Adjust layout parameters to minimize white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert plot to an image and store in session state
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    st.session_state['plot_image'] = Image.open(buf)


def display_histogram(selected_index):
    """Function to display the histogram."""
    title= f'Image {selected_index}'
    top_RGB_colors = get_colors(st.session_state['segmented_images'][selected_index], 6, True)
    color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map = calculate_distances_to_colors(image=st.session_state['segmented_images'][selected_index])
    plot_compare(st.session_state['segmented_images'][selected_index], color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map, title)
    plot_compare_mapped_image(st.session_state['segmented_images'][selected_index])

    # Define a function to be called when the model selection changes
def on_model_change():
    # Update the session state to indicate that the model has changed
    st.session_state.model_changed = True
    st.session_state['segment'] = False

# Initialize the session state
if 'model_changed' not in st.session_state:
    st.session_state['model_changed'] = False
    st.session_state['segment'] = False
        
# Streamlit app
st.title('Image Segmentation and Analysis')

        
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Process the image
    image = get_image(uploaded_file)

    # Display the image
    st.image(image, caption='Processed Image', use_column_width=True)


    # Model selection with on_change callback
    model_option = st.selectbox(
        'Select a Model for Segmentation', 
        ('Model_H', 'Model_L', 'Model_B'),
        key='model_option',
        on_change=on_model_change  # Call on_model_change when the selection changes
    )

    if  st.button('Segment Image'):
        st.session_state['segment'] = True
        # if st.session_state.get('segment'):
        masks = load_model_and_segment(image, model_option)
        list_of_images, titles = process_images(image=image, masks=masks)
        
        # Create the plot and store it in session state
        create_plot(image, masks)

        # Save the segmented images and titles in session state
        st.session_state['segmented_images'] = list_of_images
        st.session_state['titles'] = titles

        # Initial selection
        st.session_state['selected_index'] = 0

    if st.session_state.get('segment'):
        if 'plot_image' in st.session_state:
            # Display the stored plot
            st.image(st.session_state['plot_image'], caption='Original Image with Annotations')

        if 'segmented_images' in st.session_state:
            # User selects from the segmented images
            selected_index = st.selectbox('Select a segmented image', 
                                          range(len(st.session_state['segmented_images'])),
                                          key='selected_image_index')
        
            # Display the selected segmented image
            st.image(st.session_state['segmented_images'][selected_index], 
                     caption=f'Image {selected_index}',
                     use_column_width=True)
            
            # Button to trigger the histogram plot
            if st.button('Analyze colors in the selected image'):
                st.session_state['plot_histogram'] = True
        
            # Check if the histogram should be displayed
            if st.session_state.get('plot_histogram'):
                display_histogram(selected_index)
                st.session_state['plot_histogram'] = False  # Reset the flag
            