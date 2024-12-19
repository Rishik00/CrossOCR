import streamlit as st
from PIL import Image
import os

## Local imports
from ollamaclient import OllamaInput, get_ollama_output

# Title and description
st.title("Image Uploader with Message")
st.write("Upload an image, enter a message, and it will be displayed below!")

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Message input field
message = st.text_input("Enter your message")

# Send button logic
if st.button("Send"):
    # Validation
    if not uploaded_file:
        st.error("No image uploaded. Please upload an image before sending.")
    elif not message.strip():
        st.error("Message is empty. Please enter a valid message.")
    else:
        # Save the uploaded file locally
        save_path = os.path.join("uploaded_images", uploaded_file.name)
        os.makedirs("uploaded_images", exist_ok=True)  # Create the directory if it doesn't exist

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save file content to disk

        # Display the uploaded image
        image = Image.open(save_path)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        st.success(f"Image uploaded successfully! Saved at: {save_path}")

        # Display the message
        st.write(f"**Your Message:** {message}")

        # Send data to Ollama
        ollama_input = OllamaInput(query=message, image_path=save_path)
        st.info("Sending data to Ollama...")
        try:
            # Call the Ollama client and get the result
            response = get_ollama_output(model_name="llava-phi3", inputs=ollama_input)
            st.success("Data sent successfully to Ollama!")
            st.write(f"**Ollama's Response:** {response}")
        except Exception as e:
            st.error(f"An error occurred while sending data to Ollama: {str(e)}")
