import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import pickle
import itertools
import seaborn as sns

@st.cache(allow_output_mutation=True)
def load_modelobjects():
    pickle_in = open("C://mlops_plugin//src//models//pickle_temp//ModelBaseObject.pickle","rb")
    model_dict = pickle.load(pickle_in)
    return model_dict

#ModelBaseObject_featureimpdf
@st.cache(allow_output_mutation=True)
def load_modelobjects_featuredf():
    pickle_in = open("C://mlops_plugin//src//models//pickle_temp//ModelBaseObject_featureimpdf.pickle","rb")
    feat_imp_df = pickle.load(pickle_in)
    return feat_imp_df

def getModelDataframe(base_model_output):
  score_table =  pd.DataFrame.from_dict(base_model_output).T
  score_table['ModelName'] = score_table.index
  return score_table

def getFeatureImportanceGraph(ordered_feature_importance_df):
      feature_importance_df_sorted = pd.DataFrame()
      feature_importance_df_sorted.append(ordered_feature_importance_df)
      return feature_importance_df_sorted
      # fig, ax = plt.subplots(figsize=(10,7), dpi=80)
      # sns.barplot(data=ordered_feature_importance_df, y=ordered_feature_importance_df.index, x='SumofImp', palette='magma')
      # ax.spines['right'].set_visible(False)
      # ax.spines['top'].set_visible(False)
      # ax.spines['bottom'].set_visible(False)
      # ax.xaxis.set_visible(False)
      # ax.grid(False)
      # ax.set_title('Aggregated Feature Importances for Models');
      # return sns

def main():
    st.header("Base Model Analysis Dashbaord")

    data = getModelDataframe(load_modelobjects())
    st.write(data)
    try:
        MultiSelect_MetricDifference = st.multiselect("Select Train & Test Score to Check Overfitting & Underfitting - Select 2",data.columns)
        data_s = data.copy()
        data['Difference'] = data[MultiSelect_MetricDifference[0]] - data[MultiSelect_MetricDifference[1]]
        data['Difference_Bin'] = pd.cut(data.Difference,5)
        select_x = st.radio("Select X ? ",MultiSelect_MetricDifference)
        ax = plt.figure(figsize=(18,8))
        fig = px.scatter(data, x=select_x,y="ModelName", color="Difference_Bin",size='Time',hover_data=[MultiSelect_MetricDifference[0]])
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass
    st.write("If Train is More ---> Underfitting ;  Test is more --> Overfitting. ")
    ordered_feature_importance_df =load_modelobjects_featuredf()
    fig = px.bar(ordered_feature_importance_df,x="SumofImp",y=ordered_feature_importance_df.index,orientation='h')
    st.plotly_chart(fig, use_container_width=True)
  #   x_col = st.selectbox('What is your Column for X ?',data.columns)
  #   ModelDataFrame['Difference_Bin'] = pd.cut(ModelDataFrame[difference_col],Difference_bins)
  #
  #   ax = plt.figure(figsize=(18,8))
  # # sns.scatterplot(x=x_col, y="MLName",data=ModelDataFrame,size='Time', hue='Difference_Bin',sizes=(20, 600), hue_norm=(0, 20))
  #   fig = px.scatter(ModelDataFrame, x=x_col, y="MLName", color="Difference_Bin",size=size)
  #   st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
