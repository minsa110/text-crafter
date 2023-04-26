import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define a function to load the model and tokenizer
@st.cache_data(show_spinner=False)
def load_model_and_tokenizer(model_path):
    # Load the tokenizer and model from the local directory
    ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
    ft_model = AutoModelForCausalLM.from_pretrained(model_path)

    return ft_tokenizer, ft_model

# Load the model and tokenizer
model_path = "./model/medium-tech"
ft_tokenizer, ft_model = load_model_and_tokenizer(model_path)

def main():
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
