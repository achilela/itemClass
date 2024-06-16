import streamlit as st
import openai
import pandas as pd
import os

# Load the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the model ID
model_id = "ft:gpt-3.5-turbo-0125:valonylabsz:finetune-itemclass:9aIqocEw"

# Function to classify a single item
def classify_item(item):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies maintenance items."},
            {"role": "user", "content": f"Classify the following maintenance item: {item}"}
        ]
    )
    return response.choices[0].message["content"]

# Streamlit app
st.title("Item Classification with Fine-Tuned OpenAI Model")

# Single item classification
st.header("Classify a Single Item")
single_item = st.text_input("Enter the item description:")
if st.button("Classify Single Item"):
    if single_item:
        prediction = classify_item(single_item)
        st.write(f"ItemClass Prediction: {prediction}")
    else:
        st.write("Please enter an item description.")

# Batch classification
st.header("Classify Items in Tabular Form")
uploaded_file = st.file_uploader("Upload a CSV file with items to classify", type=[".csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Item' not in df.columns:
        st.error("CSV must contain an 'Item' column.")
    else:
        if st.button("Classify Items"):
            df['Prediction'] = df['Item'].apply(classify_item)
            st.write("Classification Results:")
            st.write(df)
            csv = df.to_csv(index=False)
            st.download_button(label="Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("""
    ### Usage:
    - To classify a single item, enter the description in the text box and click the "Classify Single Item" button.
    - To classify multiple items, upload a CSV file with an "Item" column and click the "Classify Items" button.
""")
