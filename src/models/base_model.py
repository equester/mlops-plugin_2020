from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
import pandas as pd
from sklearn import metrics

class BaseModellingHelper:

    def __init__(self, std_param, base_model):
        # if not set(grid_models.keys()).issubset(set(grid_params.keys()))  \
        #       or not set(popt_models.keys()).issubset(set(popt_params.keys())) \
        #       or not set(automl_model.keys()).issubset(set(automl_params.keys())) \
        #       or not set(dl_model.keys()).issubset(set(dl_params.keys())):

        #     missing_params_grid = list(set(grid_models.keys()) - set(grid_params.keys()))
        #     missing_params_popt = list(set(popt_models.keys()) - set(popt_params.keys()))
        #     missing_params_automl = list(set(automl_model.keys()) - set(automl_params.keys()))
        #     missing_params_dl = list(set(dl_model.keys()) - set(dl_params.keys()))

        #     raise ValueError("Some estimators are missing parameters: %s" % missing_params_grid, missing_params_popt,missing_params_automl,missing_params_dl)

        self.std_param = std_param
        if self.std_param['Split_type'] == 'ShuffleSplit':
          self.cross_val = model_selection.ShuffleSplit(n_splits = self.std_param['folds'], test_size = self.std_param['test_size'], train_size = self.std_param['train_size'], random_state = self.std_param['seed'] )

        self.base_model = base_model
        self.base_model_output = {}
        self.feature_importance_df_sorted = pd.DataFrame()
        self.important_col =[]

        self.scoring = { 'accuracy' : make_scorer(metrics.accuracy_score),
                  'precision' : make_scorer(metrics.precision_score),
                  'recall' : make_scorer(metrics.recall_score),
                  'f1_score' : make_scorer(metrics.f1_score),
                  'average_precision': make_scorer(metrics.average_precision_score),
                  'balanced_accuracy': make_scorer(metrics.balanced_accuracy_score),
                  'hamming_loss':make_scorer(metrics.hamming_loss),
                  'jaccard_score': make_scorer(metrics.jaccard_score),
                  'log_loss': make_scorer(metrics.log_loss),
                  'roc_auc_score':make_scorer(metrics.roc_auc_score),
                  'zero_one_loss':make_scorer(metrics.zero_one_loss,normalize=False)
                  }


        self.scores_list = []
        # self.grid_searches = {}
        # self.best_params = {}
        self.feature_importance = {}
        self.FeatureImportanceAlgo = ['DecisionTreeClassifier','RandomForestClassifier','ExtraTreesClassifier','GradientBoostingClassifier','AdaBoostClassifier']
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.std_param['test_size'])

    def getScoreDictionary(self, base_model_output,modelname, basemodel_scores, score_type):
      self.base_model_output[modelname]['Time'] = basemodel_scores['fit_time'].mean()
      self.base_model_output[modelname]['%s_accuracy' %score_type ] =  basemodel_scores['%s_accuracy' %score_type].mean()
      self.base_model_output[modelname]['%s_precision' %score_type ] =  basemodel_scores['%s_precision' %score_type].mean()
      self.base_model_output[modelname]['%s_recall' %score_type ] =  basemodel_scores['%s_recall' %score_type].mean()
      self.base_model_output[modelname]['%s_f1_score' %score_type ] =  basemodel_scores['%s_f1_score' %score_type].mean()
      self.base_model_output[modelname]['%s_average_precision' %score_type ] =  basemodel_scores['%s_average_precision' %score_type].mean()
      self.base_model_output[modelname]['%s_balanced_accuracy' %score_type ] =  basemodel_scores['%s_balanced_accuracy' %score_type].mean()
      self.base_model_output[modelname]['%s_hamming_loss' %score_type ] =  basemodel_scores['%s_hamming_loss' %score_type].mean()
      self.base_model_output[modelname]['%s_jaccard_score' %score_type ] =  basemodel_scores['%s_jaccard_score' %score_type].mean()
      self.base_model_output[modelname]['%s_log_loss' %score_type ] =  basemodel_scores['%s_log_loss' %score_type].mean()
      self.base_model_output[modelname]['%s_roc_auc_score' %score_type ] =  basemodel_scores['%s_roc_auc_score' %score_type].mean()
      self.base_model_output[modelname]['%s_zero_one_loss' %score_type ] =  basemodel_scores['%s_zero_one_loss' %score_type].mean()

      return None

    def ModelLoop(self,X, y, score_type=None):
      for key, eachModel in self.base_model.items():
          basemodel_scores = model_selection.cross_validate(eachModel, X,y, cv  = self.cross_val,return_train_score=True,scoring=self.scoring, pre_dispatch="2*n_jobs")
          modelname = eachModel.__class__.__name__
          self.base_model_output[modelname] = {}
          self.getScoreDictionary(self.base_model_output,modelname, basemodel_scores, 'train')
          self.getScoreDictionary(self.base_model_output,modelname, basemodel_scores, 'test')
          self.scores_list.append(basemodel_scores)
          if eachModel.__class__.__name__ in self.FeatureImportanceAlgo:
            eachModel.fit(X,y)
            self.feature_importance[eachModel.__class__.__name__]= eachModel.feature_importances_

    def runBaseLineModel(self, X, y, score_type=None, auto_feature_eng = None , top_feature = None ):
      if top_feature:
        print ("Building model with only %s important feature" % top_feature)
        #Initial Model Loop to extract top feature
        self.ModelLoop(X, y, score_type)
        imp_df = self.getFeatureImportance(self.getFeatureImportanceDF(X, self.feature_importance))
        important_col = list(imp_df[:top_feature].index)
        self.important_col = important_col
        X = X[important_col]
        self.ModelLoop(X, y,score_type)
      else:
        print ("Building model without any important feature")
        self.ModelLoop(X, y,score_type)

    def getFeatureImportanceDF(self, X, feature_importance_dict, important_col=None):
      if important_col:
        feature_names = important_col
        feat_imp_df = pd.DataFrame.from_dict(feature_importance_dict)
        feat_imp_df.index = feature_names
        return feat_imp_df
      else:
        feature_names = X.columns
        feat_imp_df = pd.DataFrame.from_dict(feature_importance_dict)
        feat_imp_df.index = feature_names
        return feat_imp_df

    def getFeatureImportance(self,feat_imp_df):
      mms = MinMaxScaler()
      # scaling to MinMax Scale
      scaled_fi = pd.DataFrame(data=mms.fit_transform(feat_imp_df),columns=feat_imp_df.columns,index=feat_imp_df.index)
      # Adding all values of importance to get single socre
      scaled_fi['SumofImp'] = scaled_fi.sum(axis=1)
      # print(scaled_fi.head())
      ordered_ranking = scaled_fi.sort_values('SumofImp', ascending=False)
      return ordered_ranking


    def getFeatureImportanceGraph(self,ordered_feature_importance_df):
      self.feature_importance_df_sorted.append(ordered_feature_importance_df)
      fig, ax = plt.subplots(figsize=(10,7), dpi=80)
      sns.barplot(data=ordered_feature_importance_df, y=ordered_feature_importance_df.index, x='SumofImp', palette='magma')
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.xaxis.set_visible(False)
      ax.grid(False)
      ax.set_title('Aggregated Feature Importances for Models');
      return ax

    def getModelDataframe(self, base_model_output, sort_column, asscending=False,difference_by=None, score_filter=None):
      score_table =  pd.DataFrame.from_dict(base_model_output).T
      if score_filter:
        score_columns = [['train_'+ eachScore,'test_'+eachScore] for eachScore in score_filter]
        score_columns_flat = list(itertools.chain(*score_columns))
        score_columns_flat.append("Time")
        score_table = score_table[score_columns_flat]
      score_table['Difference_%s_unit'%difference_by] = abs(score_table['train_%s' %difference_by] - score_table['test_%s' %difference_by])*100
      score_table_ordered = score_table.sort_values(sort_column, ascending=asscending)
      return score_table_ordered

    def getModelValidationGraph(self, ModelDataFrame, x_col= None, Difference_bins=5,difference_col=None,size=None):
      ModelDataFrame['MLName'] = ModelDataFrame.index
      ModelDataFrame['Difference_Bin'] = pd.cut(ModelDataFrame[difference_col],Difference_bins)
      ax = plt.figure(figsize=(18,8))
      # sns.scatterplot(x=x_col, y="MLName",data=ModelDataFrame,size='Time', hue='Difference_Bin',sizes=(20, 600), hue_norm=(0, 20))
      fig = px.scatter(ModelDataFrame, x=x_col, y="MLName", color="Difference_Bin",size=size)
      # fig.show()
      # ax.grid(False)
      # ax.set_title('Model Validation & Overfitting');
      return fig
