import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to The app! 👋")

st.sidebar.success("Select each stage above, one by one.")

st.markdown(
    """
    
# Interactive coral color analysis

This project implements the segment anything algorithm, and asses the coral heatlh accoring to the color chart given.
The project not only requires segmentation, it also applies Optical Characther Recgnition , it manipulates the image to create a Custom Color Chart 
The combination of these techniques results in the mapping of colors over the coral of interest

## Usage/Examples

Welcome to The app! 👋
This is a modular application , meaning each of the names you see on the side bar is a stage on the sequence of steps to get the result .

You are in the welcome page so lets think step by step 

### Step 1 - Cropping images 
In this part you will select two section of your images:
* the coral image
* the color chart 

After this you will save the images into the memory of the machine 

## Step 2 - Separate Colors from the chart
In this section you are required to select the color sections from the color chart (Up, Down, Left, Right)
Also this assuming that you have a color chart as in the "Screenshoot section".
#### If you need to rotate the color chart image go to the section "Rotation ofthe color chart". After you are satified with the rotation you can save the image on the memory

The sequence is as follows :
* Select up, down ... , Try to get the letter as best a posible
* Process the color crops - this will try to correct the titl on the image to make the OCR easier
* Start the OCR by pressing "Detect writting" button 
    - if the program cant detect 6 labels it will tell you 
    - you can reselect the area you have problems and process it again
* If the OCR is succesful it will deploy the custom color chart

## Step 3 - Mapping the custom color chart
In this secction the we process the coral image and finaly analize the coral Colors

Start with the segmentation of the image of the coral using the SAM model (Thanks Meta AI )
* Select the SAM model to use 
* Select the Image of the coral with black background
* Trigger the analisis byt pressing "Analyze colors in the selected image" button 
* Wait for the results

#### Interpreting the Results 
 The firts result you will get is a quick analysis of the color we can find on the coral as a pie chart that will tell you the percentaje of colors on the image
- Then it wll show the color closer to a "Ideal color chart" - meaning it calculates the color closest to the color chart what we hardcoded with the correspoing RGB values from the color chart.
- Finally the las images will be an figure that shows the original image, a mapped image generated with the colors from the custom color chart and the percentage of each custon color on the picture 
## Authors

- [Dr. Didier Barradas Bautista](https://www.github.com/D-barradas)


## Appendix

A video explainign the usage will be here but this is aplace holder for the moment


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## 🔗 Links

[KAUST Core Labs](https://corelabs.kaust.edu.sa/
) : 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/kaust-core-labs/about/) [![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/kaust_corelabs)

[KAUST Supercomputing Lab](https://www.hpc.kaust.edu.sa/) : 
[![KAUST_HPC](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/KAUST_HPC) 

[KAUST Vizualization Core Lab](https://corelabs.kaust.edu.sa/labs/detail/visualization-core-lab) :
[![KVL](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/KAUST_Vislab)  
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCR1RFwgvADo5CutK0LnZRrw?style=social)](https://www.youtube.com/channel/UCR1RFwgvADo5CutK0LnZRrw)



## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


"""
)
