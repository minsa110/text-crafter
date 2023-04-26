import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fsspec
import tempfile
import os
import shutil

# streamlit run app.py --server.enableCORS=false

# Define a function to load the model and tokenizer
@st.cache_data(show_spinner=False)
def load_model_and_tokenizer(model_path):
    # Replace the placeholders with your own values
    storage_account_name = "sominmodelstorage"
    container_name = "mediummodel"
    model_directory = "model/medium-tech"  # The directory in the container where the model files are stored

    # Create a file system that maps to Azure Blob Storage
    fs = fsspec.filesystem("az", account_name=storage_account_name)

    # Create a temporary directory to store the downloaded model files
    temp_dir = tempfile.mkdtemp()

    # Download all model files from the container to the temporary directory
    for file in fs.glob(f"az://{container_name}/{model_directory}/*"):
        with fs.open(file, "rb") as src, open(os.path.join(temp_dir, os.path.basename(file)), "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Load the tokenizer and model from the temporary directory
    ft_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
    ft_model = AutoModelForCausalLM.from_pretrained(temp_dir)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    return ft_tokenizer, ft_model

# Load the model and tokenizer
ft_tokenizer, ft_model = load_model_and_tokenizer("https://sominmodelstorage.blob.core.windows.net/mediummodel/model%2Fmedium-tech%2F")

def main():
    # st.title("California Housing Prediction")
    html_title = """
    <div style="background:#5dc9c6 ;padding:10px">
    <h2 style="color:white;text-align:center">Medium post opinions</h2>
    </div>

    <p>Start a sentence to have Medium posts complete your sentence 🤔</p>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    # Create a text input field for user input
    text = st.text_input("Enter text:")

    # Generate response when the "Generate" button is clicked
    if st.button("Generate"):
        with st.spinner("Generating..."):
            # Tokenize the input text
            ft_input_ids = ft_tokenizer.encode(text, return_tensors='pt')
            # Generate output using the model
            output = ft_model.generate(ft_input_ids, attention_mask=torch.ones_like(ft_input_ids),
                                       pad_token_id=ft_tokenizer.eos_token_id,
                                       max_length=100, do_sample=True)
            # Decode and display the output
            response = ft_tokenizer.decode(output[0], skip_special_tokens=True)
            st.write(response)

# Run the app
if __name__ == "__main__":
    main()
