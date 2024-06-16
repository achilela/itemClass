import streamlit as st
import openai
import pandas as pd
from openai import OpenAI
client = OpenAI(api_key = "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")


# Set your OpenAI API key
#openai.api_key = "YOUR_OPENAI_API_KEY"

# Model ID
model_id = "ft:gpt-3.5-turbo-0125:valonylabsz:finetune-itemclass:9aIqocEw"

def Inference_func(Prompt, question, model):
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": Prompt},
                {"role": "user", "content": question},
            ]
        )
        output = response.choices[0].message.content


        
        return output
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None

st.title("Text Classification with OpenAI Fine-Tuned Model")

st.write("This app classifies input text using a fine-tuned GPT-3.5 model.")

# Option to choose between single input and tabular input
input_type = st.radio("Choose input type:", ("Single Input", "Tabular Input"))

if input_type == "Single Input":
    single_input = st.text_area("Enter text to classify:")
    if st.button("Classify"):
        if single_input:
            result = Inference_func("Please classify the following text:", single_input, model_id)
            if result is not None:
                st.write("Classification Result:", result)
            else:
                st.write("Failed to classify the text. Please check the error message above.")
        else:
            st.write("Please enter some text to classify.")

elif input_type == "Tabular Input":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)
        
        if st.button("Classify"):
            df['Classification'] = df.iloc[:, 0].apply(lambda x: Inference_func("Please classify the following text:", x, model_id))
            st.write("Classification Results:")
            st.write(df)

