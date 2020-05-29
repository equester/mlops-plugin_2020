import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import pickle
import itertools
import shap
import xgboost

st.header("Model Interpretation")
models_names = ["model-ExtraTreesClassifier0.517597",
 "model-XGBClassifier0.239277",
 "model-BaggingClassifier0.413186",
 "model-DecisionTreeClassifier0.540355",
 "model-RandomForestClassifier0.870131" ,
 "model-LGBMClassifier0.357780",
 "model-GradientBoostingClassifier0.174659"]
option = st.selectbox("Select Model to Study", models_names)
path = "C://mlops_plugin//models//%s//model.pkl" %option
model = pickle.load(open(path, 'rb'))
# model = pickle.load(open("C://mlops_plugin//models//model-XGBClassifier0.239277//model.pkl", 'rb'))
X = pd.read_csv("C:/mlops_plugin/src/data/x_train.csv")

st.header("Global explication")
st.write("""In the following figure, each point represents a passenger, the horizontal axis is SHAP value (the contribution of this feature to the survival probability). The color of the point indicates the feature value across the entire feature range. For example, the blue points in age represent children and the red point is elder. The features are sorted from top to bottom in order of decreasing importance. From the figure we can see some results that match the intuition such as females (red) have survival advantage; the higher the class, the more survival advantage; children (blue) have more survival advantage. We find also some interesting phenomena like: people with family members have a survival advantage over alone, but if there are too many family members, it is worse than alone.""")
#Global
explainer = shap.TreeExplainer(model, model_output='margin')
shap_values = explainer.shap_values(X)
df_shap = X.copy()
df_shap.loc[:,:] = shap_values
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
plt.clf()

st.header("Feature dependence explication")
st.write("""Shap also support plot cross-influence from two features. From the Following dependence figure, we find that from the first class to the third class, both males (blue) and females (red), their survival advantages have been weakened, but this weakening is particularly evident in women. We can think that because the male survival rate is already low, the impact of class on male survival is not so sensible. However, the female survival rate is much higher, therefore, whether the woman is in the third-class becomes an important condition for females' survival. So we can say that the surviving bias due to the class is more pronounced in females.""")

for feat in df_shap.columns:
    shap.dependence_plot(feat, shap_values, X, interaction_index='auto')
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
