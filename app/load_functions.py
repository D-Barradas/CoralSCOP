import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import os 
from glob import glob 
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import StringIO


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def preprocess_histograms(image):
    #seperating colour channels
    B = image[:,:,0] #blue layer
    G = image[:,:,1] #green layer
    R = image[:,:,2] #red layer
    # equalize the histograms 
    b_equi = cv2.equalizeHist(B)
    g_equi = cv2.equalizeHist(G)
    r_equi = cv2.equalizeHist(R)
    equi_im = cv2.merge([b_equi,g_equi,r_equi])
    return equi_im

# Modify the get_image function to work with Streamlit's uploaded file
def get_image(uploaded_file):
    # Read the file into a byte stream
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode the file bytes to an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply any preprocessing like histogram preprocessing
    image = preprocess_histograms(image=image)

    # Resize the image
    image = cv2.resize(image, (1200, 800), interpolation=cv2.INTER_AREA)
    
    return image

def show_images_grid(images, titles=None, figsize=(20, 20)):
    """Displays a grid of images with optional titles."""

    num_images = len(images)
    rows = int(num_images / 2)
    cols = 2

    # Create a figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten the subplots array for easier iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            ax.imshow(img)
            # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for Matplotlib
            ax.axis('off')  # Hide axes

            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    plt.show()
def background_to_black ( image, index ):
    # Apply the mask to the image
    masked_img = image.copy()
    masked_pixels = masked_img[masks[index]['segmentation']==True]
    masked_img[masks[index]['segmentation']==False] = (0, 0, 0)  # Set masked pixels to black
    return masked_img ,masked_pixels

def get_sorted_by_area(image, anns):
    area_list=[]
    cropped_image_dic ={}
    mask_number = [] 
    mask_pixles_dic = {}
    for i in range(len(anns)):
        x, y, width, height = anns[i]['bbox']
        area = anns[i]["area"]
        image_b, masked_pixels = background_to_black(image=image, index=i)
        cropped_image = image_b[int(y):int(y+height), int(x):int(x+width)]

        area_list.append(area)
        cropped_image_dic[i] = cropped_image
        mask_pixles_dic[i] = masked_pixels
        mask_number.append(i)
    df = pd.DataFrame([area_list,mask_number])
    df = df.T
    df.columns = ['area','mask_number']
    df.sort_values(by='area', ascending=False, inplace=True)
    df.dropna(inplace=True)
    return df , cropped_image_dic , mask_pixles_dic




def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colors(image, number_of_colors, show_chart):
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors, n_init='auto', random_state=73)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()), 
                     color_discrete_sequence=hex_colors,
                     title="Color Distribution")

        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

    return rgb_colors
    


def drop_black_from_top_colors(top_colors_list):
    min_values = []
    for i in range(len(top_colors_list)):
        curr_color = rgb2lab(np.uint8(np.asarray([[top_colors_list[i]]])))
        diff = deltaE_cie76((0, 0, 0), curr_color)
        # print (diff, type(diff))
        min_values.append(diff[0][0])
        lowest_value_index = np.argmin(min_values) 
    top_colors_list.pop(lowest_value_index)
    return top_colors_list


def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    # discard black
    image_colors = drop_black_from_top_colors(image_colors)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    diff_list =[]
    for i in range(len(image_colors)):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        # print(diff[0][0])
        diff_list.append(diff[0][0])
    diff_avg = np.mean(diff_list)
    if diff_avg < threshold:
        return diff_avg
    else:
        # a euclidian difference of 1000 should be noticible 
        return 1000
    
def calculate_distances_to_colors(image):
    # color chart but in RGB 
    color_map_RGB = {
    'B1': (247, 248, 232),
    'B2': (243, 244, 192),
    'B3': (234, 235, 137),
    'B4': (200, 206, 57),
    'B5': (148, 157, 56),
    'B6': (92, 116, 52),
    'C1': (247, 235, 232),
    'C2': (246, 201, 192),
    'C3': (240, 156, 136),
    'C4': (207, 90, 58),
    'C5': (155, 50, 32),
    'C6': (101, 27, 13),
    'D1': (246, 235, 224),
    'D2': (246, 219, 191),
    'D3': (239, 188, 135),
    'D4': (211, 147, 78),
    'D5': (151, 89, 36),
    'D6': (106, 58, 22),
    'E1': (247, 242, 227),
    'E2': (246, 232, 191),
    'E3': (240, 213, 136),
    'E4': (209, 174, 68),
    'E5': (155, 124, 45),
    'E6': (111, 85, 34)
    }
    
    # get the distance 
    final_distances = {}
    for key in color_map_RGB.keys():
        max_val = match_image_by_color( image=image, color=color_map_RGB[key], number_of_colors=6)
        if max_val != 0 :
            final_distances[key]=max_val
    df_final = pd.DataFrame.from_dict(final_distances,orient='index',columns=["Distance"])
    df_final.sort_values(by="Distance",ascending=True,inplace=True)
    color_keys_selected= df_final.head(n=3).index.to_list()
    color_selected_distance = df_final["Distance"].head(n=3).to_list()
    lower_y_limit = color_selected_distance[0] - 0.5
    higher_y_limit = color_selected_distance[-1] + 0.5
    hex_colors_map = [RGB2HEX(color_map_RGB[key]) for key in color_keys_selected]
    return color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit,hex_colors_map


def plot_compare(img1_rgb, color_keys_selected, color_selected_distance, lower_y_limit, higher_y_limit, hex_colors_map,title):
    # Create a subplot grid with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title, "Euclidean Distance from Top 5 Colors Detected"))

    # Add the image to the first subplot
    fig.add_trace(go.Image(z=img1_rgb), row=1, col=1)

    # Add the bar chart to the second subplot
    fig.add_trace(go.Bar(x=color_keys_selected, y=color_selected_distance, marker_color=hex_colors_map), row=1, col=2)

    # Update y-axis range for the bar chart
    fig.update_yaxes(range=[lower_y_limit, higher_y_limit], row=1, col=2)

    # Update layout and axis properties
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Color code in chart", row=1, col=2)
    fig.update_yaxes(title_text="Euclidean Distance", row=1, col=2)
    
    # Adjust layout for a better fit
    fig.update_layout(height=600, width=1100)

    # Show the plot
    st.plotly_chart(fig)
    #fig.show()



def process_images(image, masks):
    image_dataframe, cropped_image_list , mask_pixels_dict = get_sorted_by_area( image=image , anns=masks )
    top_six_img_by_area = image_dataframe['mask_number'].head(n=10).to_list()
    list_of_images = [ cropped_image_list [idx ] for idx in top_six_img_by_area  ]
    titles = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6','Image 7','Image 8','Image 9','Image 10']
    return list_of_images , titles












# def map_color_to_pixels(image):
#    # in this color map I added black and white to conserve the black 
#    color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
#                     'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
#                     'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
#                     'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}
#    palette = [ rgb2lab(np.uint8 ( np.asarray ( color_map_RGB[x] ))) for x in color_map_RGB.keys() ]
#    palette_keys = [ x for x in color_map_RGB.keys() ]
#    mapped_img = np.zeros_like(image)
#    for i in range(image.shape[0]):
#      for j in range(image.shape[1]):
#        pixel = rgb2lab(np.uint8 ( np.asarray ( image[i, j] ) ) )
#        distances = deltaE_cie76 ( lab1=pixel , lab2=palette )
#        closest_index = np.argmin(distances)  # Find closest color index
# # assing the color to the pixel in the mapped image
#        mapped_img[i, j] = color_map_RGB[palette_keys[closest_index]]

#    return mapped_img , color_map_RGB

def closest_color(pixel, palette, palette_keys, color_map_RGB):
    pixel_lab = rgb2lab(np.uint8(pixel))
    distances = deltaE_cie76(pixel_lab, palette)
    closest_index = np.argmin(distances)
    return color_map_RGB[palette_keys[closest_index]]

def map_color_to_pixels(image):
    color_map_RGB = {'Black':(0,0,0),'White':(255,255,255),'B1': (247, 248, 232),'B2': (243, 244, 192),'B3': (234, 235, 137),'B4': (200, 206, 57),'B5': (148, 157, 56),
                    'B6': (92, 116, 52),'C1': (247, 235, 232),'C2': (246, 201, 192),'C3': (240, 156, 136),'C4': (207, 90, 58),'C5': (155, 50, 32),'C6': (101, 27, 13),
                    'D1': (246, 235, 224),'D2': (246, 219, 191),'D3': (239, 188, 135),'D4': (211, 147, 78),'D5': (151, 89, 36),'D6': (106, 58, 22),'E1': (247, 242, 227),
                    'E2': (246, 232, 191),'E3': (240, 213, 136),'E4': (209, 174, 68),'E5': (155, 124, 45),'E6': (111, 85, 34)}
    
    # Convert the colors in the color map to LAB space
    palette = np.array([rgb2lab(np.uint8(np.asarray(color_map_RGB[key]))) for key in color_map_RGB.keys()])
    palette_keys = list(color_map_RGB.keys())

    # Function to apply to each pixel
    func = lambda pixel: closest_color(pixel, palette, palette_keys, color_map_RGB)

    # Apply the function to each pixel
    mapped_img = np.apply_along_axis(func, -1, image)

    return mapped_img ,color_map_RGB

from collections import Counter

def count_pixel_colors(image, color_map_RGB):
  """
  Counts the number of pixels of each color in an image.

  Args:
    image: A NumPy array representing the image.
    color_map_RGB: A dictionary mapping color names to RGB tuples.

  Returns:
    A dictionary mapping color names to the number of pixels of that color in the image.
  """
  # Flatten the image into a 1D array
  # image_flat = image.flatten()
  # return image_flat
  reverse_dict = { value : key for key , value in color_map_RGB.items() }  


  # iterate over the image pixels
  all_pixels_list =[]
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        pixel = image[i, j]  
        # discard black 
        # if reverse_dict[str(pixel)] != 'Black':
        all_pixels_list.append(pixel)

  # # Count the occurrences of each pixel value
  pixel_counts = Counter(tuple(pixel_1) for pixel_1 in all_pixels_list)
  # delete the black key from the dictionary 
  del pixel_counts[(0,0,0)] 

  # pass the values to a list 
  total_pixels = [ item for key , item in pixel_counts.items() if key != (0,0,0)]
  # sum all the values 
  total_pixels = np.sum(total_pixels)
  # # Count the number of pixels of each color in the color map
  color_counts = {color_name: pixel_counts.get(color_rgb, 0)/total_pixels * 100 for color_rgb,color_name in reverse_dict.items()}

  return pixel_counts, color_counts 



def plot_compare_mapped_image(img1_rgb):
    # Assuming map_color_to_pixels and count_pixel_colors are defined elsewhere
    mapped_image, color_map = map_color_to_pixels(image=img1_rgb)
    del color_map['Black']

    color_counts, reverse_dict = count_pixel_colors(image=mapped_image, color_map_RGB=color_map)
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
    )

    




