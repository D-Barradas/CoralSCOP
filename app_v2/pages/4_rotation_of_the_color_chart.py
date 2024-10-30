import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page


def rotate_image(image, degrees):
    """Rotates an image using Pillow and OpenCV.

    Args:
        image: The PIL Image object to rotate.
        degrees: The rotation angle in degrees.

    Returns:
        The rotated PIL Image object.
    """

    # Convert the PIL Image to a NumPy array for OpenCV
    img_array = np.array(image)

    # add padding to the image to avoid cropping
    img_array = cv2.copyMakeBorder(img_array, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Rotate the image using OpenCV's rotation matrix
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated_img_array = cv2.warpAffine(img_array, matrix, (width, height))

    # Convert the rotated NumPy array back to a PIL Image
    # rotated_image = Image.fromarray(rotated_img_array)

    return rotated_img_array


def switch_to_color():
    """Must be one of ['streamlit starting page', 'cropping color chart', 'separate color chart', 'analysis and mapping', 'rotation of the color chart']"""

    want_to_contribute = st.button("Go back to Separate the color chart segments?")
    if want_to_contribute:
        switch_page("separate color chart")


# ... (code to display the rotated image)
def main():
    image = st.session_state["chart_img"]
    # ... (code to create a slider for rotation)
    rotation_angle = st.slider("Rotate image:", min_value=-180, max_value=180)

    # ... (code to apply the rotation)
    rotated_image = rotate_image(st.session_state["chart_img"], rotation_angle)
    st.image(rotated_image)
    if st.button("Save Images to memory"):
        st.session_state["chart_img"] = rotated_image
    switch_to_color()

# Streamlit app execution
if __name__ == '__main__':
    main()
