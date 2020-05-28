import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv("C://mlops_plugin//src//visualization//required_data//basemodel_output.csv")

def main():
    data = load_data()
    # Plot All Accuracy
    difference_col ="Difference_recall_unit"
    Difference_bins = 5
    size = 'Time'
    # x_col='train_recall'
    # option 1 - What is your Column for X

    ModelDataFrame = data.copy()
    ModelDataFrame['MLName'] = data.index
    st.write(ModelDataFrame)
    x_col = st.selectbox('What is your Column for X ?',data.columns)
    ModelDataFrame['Difference_Bin'] = pd.cut(ModelDataFrame[difference_col],Difference_bins)

    ax = plt.figure(figsize=(18,8))
  # sns.scatterplot(x=x_col, y="MLName",data=ModelDataFrame,size='Time', hue='Difference_Bin',sizes=(20, 600), hue_norm=(0, 20))
    fig = px.scatter(ModelDataFrame, x=x_col, y="MLName", color="Difference_Bin",size=size)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
