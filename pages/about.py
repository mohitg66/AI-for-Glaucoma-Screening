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
- Color Fundus Photographs (CFPs) and Optical Coherence Tomography (OCT) are used for detection of Glaucoma.

**AI-based Screening**:
- Promising results in AI-based screening.
- Need for robust and generalizable algorithms.
""")

st.write("""
## Developing a Robust Glaucoma Detection Model
#### Dataset
- **Size**: Over 101,000 annotated Color Fundus Photographs (CFPs).
- **Labels**: "Referable Glaucoma" (RG) and "No Referable Glaucoma" (NRG).
- **Dataset Link**: https://zenodo.org/records/10035093
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
- Didn't use Data Augmentation to focus on the central areas using Cropping.
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
#### Model Architecture
- Convolutional Neural Network (CNN) with Custom Dense Layers.
- Transfer learning using VGG16 for feature extraction.
- Dropout layers for regularization.
- Binary cross-entropy loss function and Adam optimizer.
""")

data_model = {
    "Layers": [
        "Input Layer (120,200)", "VGG16", "Flatten", 
        "Dense (128)", "Dropout (0.1)", "Dense (128)", 
        "Dropout (0.1)", "Dense (1)"
    ]
}

# Create a DataFrame
df_model = pd.DataFrame(data_model)
df_model_horizontal = df_model.T

# Create a table
st.write('')
st.table(df_model_horizontal)

# AUC - 92.4%
# image roc_curve
st.write("""
#### Evaluation Metrics
- Used AUC as the primary evaluation metric as it is robust to Class Imbalance.
- Achieved AUC of **92.4%** on the validation set.
""")
st.image("images/roc_curve.png", caption='ROC Curve')
