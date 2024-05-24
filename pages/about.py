import streamlit as st
import pandas as pd
import altair as alt
from menu import menu
menu()

# def app():
st.title('About the Project')
st.markdown(""" 
## Addressing the Challenge of Glaucoma
**Glaucoma Prevalence**:
- One of the leading causes of blindness.
- Early diagnosis is crucial to prevent progression.          

**Imaging Techniques**:
- Color fundus photographs (CFPs) and optical coherence tomography (OCT) are used for detection of Glaucoma.

**AI-based Screening**:
- Promising results in AI-based screening.
- Need for robust and generalizable algorithms.
""")

st.write("""
## Developing a Robust Glaucoma Detection Model
#### Dataset
- **Size**: Over 101,000 annotated CFPs.
- **Labels**: "Referable Glaucoma" (RG) and "No Referable Glaucoma" (NRG).
""")

# create a dataframe for dataset chart
df_dataset = pd.DataFrame({
    'Label': ['RG', 'NRG'],
    'Count': [3270, 98153]
})

# create a bar chart
st.write('')
chart = alt.Chart(df_dataset).mark_bar().encode(
    x='Label',
    y='Count',
    color=alt.Color('Label', legend= None),
).properties(
    width=500,
    height=300,
)
st.altair_chart(chart)

st.write("""
#### Preprocessing
- Cropping technique to focus on central areas.
- Grayscale conversion to emphasize textural changes.
""")

st.image("images/image1.png", caption='Before Preprocessing')
st.image("images/image2.png", caption='After Preprocessing')

st.write("""
#### Class Imbalance
- Applied oversampling on the minority class and undersampling on the majority class.
""")

df_imbalance = pd.DataFrame({
    'Label': ['RG', 'NRG'],
    'Count': [15000, 60000]
})

chart = alt.Chart(df_imbalance).mark_bar().encode(
    x='Label',
    y='Count',
    color=alt.Color('Label', legend=None),
).properties(
    width=500,
    height=200,
)
st.altair_chart(chart)

st.write("""
#### Model Architectures
- Custom CNN architectures.
- Integration of pre-trained models like VGG16.
""")

data_model = {
    "Layer": [
        "input layer (120,200)", "vgg16", "flatten", 
        "dense (128)", "dropout (0.1)", "dense (128)", 
        "dropout (0.1)", "dense (1)"
    ]
}

# Create a DataFrame
df_model = pd.DataFrame(data_model)
df_model_horizontal = df_model.T

# Create a table
st.write('')
st.table(df_model_horizontal)

st.write("""
#### Evaluation Metrics
- **Primary metric**: Area Under the Curve (AUC).
""")

# st.image('https://miro.medium.com/max/1400/0*zq9eSl7Sbt5Y7CzP.jpg', caption='Convolutional Neural Network Architecture')
