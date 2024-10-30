import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page
import sys ,os
from io import BytesIO
from zipfile import ZipFile
sys.path.append('../')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

with open("load_functions.py") as f:
    exec(f.read())


def switch_to_color():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("separate color chart")


def switch_to_manual():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to manual selection of colors?")
    if want_to_contribute:
        switch_page("manual selection of colors")



# if the color chart is not on session state ask the user to go to page 2 or 5  st.session_state["custom_color_chart"] 
def is_color_chart_in_session_state():
    if "custom_color_chart" not in st.session_state:

        st.write("Please go to page 2 or 5 to upload the color chart image")
        switch_to_color()
        switch_to_manual()

    else:
        st.write("Color chart image is already in session state")
        OcrAnalysis.plot_custom_colorchart(st.session_state["custom_color_chart"])
         


def is_cuda_available():
    """Checks if CUDA is available and can be used by PyTorch.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """

    return torch.cuda.is_available()


# Function to load a model based on selection
def load_model(model_option='Model_B'):
    sam_checkpoint = "../checkpoints/vit_b_coralscop.pth"  # this is coralSCOPE
    model_type = "vit_b"

    if is_cuda_available():
        st.markdown("CUDA is available!")
        device = torch.device("cuda:1")  # reactivate the previous line for the app
    else:
        st.markdown("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Parameters for CoralScope
    points_per_side = 32
    pred_iou_thresh = 0.72
    stability_score_thresh = 0.62

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    # masks = mask_generator.generate(image)
    return mask_generator



def plot_compare_mapped_image_batch_mode(img1_rgb,color_map_RGB, idx):
    # check if the black color is in the color map if not add it
    if 'Black' not in color_map_RGB.keys():
        color_map_RGB['Black'] = tuple([0,0,0])

    mapped_image , color_map , color_to_pixels = map_color_to_pixels(image=img1_rgb, color_map_RGB=color_map_RGB )
    del color_map['Black'] 
    del color_to_pixels['Black']


    color_counts, reverse_dict = count_pixel_colors(image=mapped_image , color_map_RGB=color_map)
    lists = sorted(reverse_dict.items(), key=lambda kv: kv[1], reverse=True)

    color_name, percentage_color_name = [], []
    for c, p in lists:
        if p > 1:
            color_name.append(c)
            percentage_color_name.append(p)

    hex_colors_map = [RGB2HEX(color_map[key]) for key in color_name]


    # Create a subplot grid with adjusted row widths and column widths
    fig = make_subplots(rows=1, cols=3, 
                        column_widths=[0.35, 0.35, 0.3],  # Adjust column widths
                        subplot_titles=("Original", "Mapped Image", "Color Distribution"))

    # Add the original image, mapped image, and the bar chart to respective subplots
    fig.add_trace(go.Image(z=img1_rgb), row=1, col=1)
    fig.add_trace(go.Image(z=mapped_image), row=1, col=2)
    fig.add_trace(go.Bar(x=color_name, y=percentage_color_name, marker_color=hex_colors_map), row=1, col=3)

    # Update layout and axis properties
    fig.update_layout(showlegend=False, height=600, width=1500)  # Adjust the total figure size
    fig.update_xaxes(title_text="Color code in chart", row=1, col=3)
    fig.update_yaxes(title_text="Percentage of pixel on the image", row=1, col=3)

    # Convert the color distribution data into a DataFrame
    color_distribution_data = pd.DataFrame({
        'Color Name': color_name,
        'Percentage': percentage_color_name,
        'Hex Color': hex_colors_map
    })
    
    # Convert the DataFrame to a CSV string
    csv = color_distribution_data.to_csv(index=False).encode('utf-8')
    
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Create a download button and offer the CSV string for download
    st.download_button(
        label="Download Color Distribution Data",
        data=csv,
        file_name="color_distribution_data.csv",
        mime="text/csv",
        key=idx
    )

def plot_compare_mapped_image_batch_mode_results_to_memory(img1_rgb, color_map_RGB, idx):
    # check if the black color is in the color map if not add it
    if 'Black' not in color_map_RGB.keys():
        color_map_RGB['Black'] = tuple([0, 0, 0])

    mapped_image, color_map, color_to_pixels = map_color_to_pixels(image=img1_rgb, color_map_RGB=color_map_RGB)
    del color_map['Black']
    del color_to_pixels['Black']

    color_counts, reverse_dict = count_pixel_colors(image=mapped_image, color_map_RGB=color_map)
    lists = sorted(reverse_dict.items(), key=lambda kv: kv[1], reverse=True)

    color_name, percentage_color_name = [], []
    for c, p in lists:
        if p > 1:
            color_name.append(c)
            percentage_color_name.append(p)

    hex_colors_map = [RGB2HEX(color_map[key]) for key in color_name]

    # Create a subplot grid with adjusted row widths and column widths
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Add the original image, mapped image, and the bar chart to respective subplots
    axes[0].imshow(img1_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(mapped_image)
    axes[1].set_title("Mapped Image")
    axes[1].axis('off')

    axes[2].bar(color_name, percentage_color_name, color=hex_colors_map)
    axes[2].set_title("Color Distribution")
    axes[2].set_xlabel("Color code in chart")
    axes[2].set_ylabel("Percentage of pixel on the image")

    plt.tight_layout()

    # Convert the color distribution data into a DataFrame
    color_distribution_data = pd.DataFrame({
        'Color Name': color_name,
        'Percentage': percentage_color_name,
        'Hex Color': hex_colors_map
    })

    # Convert the DataFrame to a CSV string
    csv = color_distribution_data.to_csv(index=False).encode('utf-8')

    return fig, csv


def main():
    is_color_chart_in_session_state()
    uploaded_files = st.file_uploader("Choose the images ...", type=["bmp", "jpg", "jpeg", "png", "svg"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        # bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)

    if st.button("Start Segmentation"):
        mask_generator = load_model()
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for idx , uploaded_file in enumerate ( uploaded_files ) :
            print (idx)
            name = uploaded_file.name.split(".")[0]

            percent_complete = (idx + 1) / len(uploaded_files)
            my_bar.progress(percent_complete , text=progress_text)

            custom_color_chart = st.session_state["custom_color_chart"]
            # print (custom_color_chart.keys() ,"for loop")
            # if idx > 0 :
            #     #add black to the custom color chart
            #     custom_color_chart['Black'] =tuple([0,0,0])

            # for each image in the uploaded files we will apply the same process 
            # use get_image function to get the image
            # then use load_model_and_segment
            # then process_images 

            image = get_image(uploaded_file)
            masks = mask_generator.generate(image)
            # at this point we have the masks and the image crops 
            list_of_images, titles = process_images(image, masks)
            # now we will plot the images
            # plot_compare_mapped_image_batch_mode(list_of_images[0],custom_color_chart,idx)
            fig , csv = plot_compare_mapped_image_batch_mode_results_to_memory(list_of_images[0], custom_color_chart, idx)
            # save fig and csv into a dictionary that dictionary will be saved in the session state
            st.session_state[f"mapped_image_{idx}_{name}"] = fig 
            st.session_state[f"color_distribution_data_{idx}_{name}"] = csv

            if len(list_of_images) > 1:
                st.write(f"Warning More than one coral image detected on image:{name}")
                # for img in list_of_images[1:]:
                #     plot_compare_mapped_image_batch_mode(img,custom_color_chart,idx)
            # for img in list_of_images:
                # plot_compare_mapped_image_batch_mode(img,custom_color_chart,idx)
    # here there is a button to download the results
    if st.button("Download Results"):
        # lets create a loop to download all the images
        # Lets put all the images in a list and create a zip file
        # lets put all the csv in a list and create a zip file
        # then we will download the zip files
        images = []
        csvs = []
        for idx, uploaded_file in enumerate(uploaded_files):
            name = uploaded_file.name.split(".")[0]
            images.append(st.session_state[f"mapped_image_{idx}_{name}"])
            csvs.append(st.session_state[f"color_distribution_data_{idx}_{name}"])

        # Create a zip file with the images
        images_zip = BytesIO()
        with ZipFile(images_zip, 'w') as z:
            for idx, image in enumerate(images):
                # buf = BytesIO()
                image_path = f"mapped_image_{idx}_{name}.png"
                image.savefig(image_path, format='png')
                z.write(filename=image_path)
                os.remove(image_path)

        # Create a zip file with the CSVs
        csvs_zip = BytesIO()
        with ZipFile(csvs_zip, 'w') as z:
            for idx, csv in enumerate(csvs):
                # print (type(csv))
                csv_path = f"color_distribution_data_{idx}_{name}.csv"
                z.writestr(csv_path, csv)
        #         os.remove(csv_path)
        
        # Download the images zip files
        st.download_button(
            label="Download Images zip file",
            data=images_zip.getvalue(),
            file_name="mapped_images.zip",
            mime="application/zip"
        )

        # Download the csv zip files
        st.download_button(
            label="Download CSVs zip file",
            data=csvs_zip.getvalue(),
            file_name="csvs_files.zip",
            mime="data/zip"
        )



            



# Streamlit app execution
if __name__ == '__main__':
    main()

 