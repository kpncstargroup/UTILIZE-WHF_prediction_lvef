################################################
#
# @Organization - Kaiser Permanente Northern California, Division of Research
#
# @Project - Developing Clinical Risk Prediction Models for Worsening Heart Failure Events 
#            and Death by Left Ventricular Ejection Fraction 
#
# @Description - This file develops the prediction models for each outcome by LVEF category 
#                in the training sets and generates performance metrics with bootstrapped 
#                95% confidence intervals in the test set. It also calculates variable importance
#                values and Shapley values for each model and outcome. Models use the H2O package.
#
# @Output - This program should output subfolders in the specified filepath for each HF subgroup 
#           and outcome (i.e. ~/filepath/ref_whf_outcome) that contain the H2O model object (MOJO),
#           and SHAP values. A file for each HF subgroup will be output in the parent folder (i.e., ~/filepath) that
#           contains the performance metrics and confidence intervals for each outcome. 
#
# @Data - The data referenced in this file is not publicly available, due to the inclusion of
#         Protected Health Information and risks of re-identification. Placeholder filepaths
#         are included in this program.    
################################################

############ Packages ############
#Load Packages
library(dplyr)
library(h2o)
library(boot)


############ Data setup ############

#Initialize H2O cluster
h2o.init(nthreads=32,ip = "localhost",port = 54323)

#Import cvs to R environment

ref_train = read_csv("~/filepath/hfref_train.csv")
ref_test = read_csv("~/filepath/hfref_test.csv")

mref_train = read_csv("~/filepath/hfmref_train.csv")
mref_test = read_csv("~/filepath/hfmref_test.csv")

pef_train = read_csv("~/filepath/hfpef_train.csv")
pef_test = read_csv("~/filepath/hfpef_test.csv")

all_train = read_csv("~/filepath/all_train.csv")
all_test = read_csv("~/filepath/all_test.csv")

trainl = list(ref_train,mref_train,pef_train,all_train)
testl = list(ref_test,mref_test,pef_test,all_test)
names(trainl) = c("ref_train","mref_train","pef_train","all_train")
names(testl) = c("ref_test","mref_test","pef_test","all_test")

#Clean data
#################

# Categorical variable list (including binary outcomes)- sex, race, tobacco use, alcohol use, drug use, 
# aortic stenosis severity, left ventricular hypertrophy, urine dipstick proteinuria, outcomes
catvars = c('demo_sex','demo_race','shx_tobacco_use','shx_alcohol_use','shx_illicit_drug_use','shx_iv_drug_use'
            ,'echo_as_severity_c','echo_lvh_c','lab_dip_c','whf_outcome','death_outcome','whf_outcome_ip',
            'whf_outcome_iped','whf_outcome_av','whf_outcome_edo')

#Convert categorical variables to factors
clean_df <- function (dsn) {
  dsn[catvars] = lapply(dsn[catvars],as.factor)
  return(dsn)
}

train_clean = lapply(trainl,clean_df)
test_clean = lapply(testl,clean_df)


#Save training data as H2O frames
h2olist = lapply(train_clean,as.h2o)
list2env(h2olist,globalenv())
list2env(test_clean,globalenv())
rm(test_clean,testl,train_clean,trainl,check)

#Variables and groups
################

#Identify predictors
x = setdiff(names(ref_train), c('death_outcome','whf_outcome','whf_outcome_ip','whf_outcome_iped','whf_outcome_edo','whf_outcome_av'))

#Identify outcomes
outcomes = list('whf_outcome','whf_outcome_ip','whf_outcome_edo','whf_outcome_av','death_outcome')
names(outcomes) = outcomes

# Xgboost hyperparamters- large search space
xgb_params = list(learn_rate = c(0.01, 0.1, 0.01),
                  max_depth = seq(1, 20),
                  sample_rate = seq(0.3, 1, 0.05),
                  col_sample_rate = seq(0.3, 1, 0.05),
                  ntrees = seq(100, 1000, 100))

#Random search over the large space- set 10 min, 100 models, or stopping tolerance
search_criteria = list(strategy = "RandomDiscrete",
                       max_runtime_secs = 600,
                       max_models = 100,
                       stopping_metric = "AUC",
                       stopping_tolerance = 0.002,
                       stopping_rounds = 5,
                       seed = 1)

### Modeling - by HF subgroup and outcome ###
################

#Function for modeling by outcomes
seq_out <- function (index,oclist,train,test,prefix) {
  
  #Set outcome
  y = unname(unlist(oclist[index]))
  
  #Set grid search id for easier referencing
  gridid = paste0("xgb_grid","_",y,"1")
  
  #Perform the grid search
  xgb_grid = h2o.grid("xgboost", x = x, y = y,
                      grid_id = gridid,
                      training_frame = train,
                      nfolds=5,
                      seed = 1,
                      hyper_params = xgb_params,
                      search_criteria = search_criteria,
                      parallelism = 0)

  # Get the grid results, sorted by AUC
  xgb_gridperf2 = h2o.getGrid(grid_id = gridid,
                              sort_by = "auc",
                              decreasing = TRUE)

  #Pull out the best model
  best_xgb = h2o.getModel(xgb_gridperf2@model_ids[[1]])
  
  #Save model to filepath
  f = paste0(prefix,"_",y)
  h2o.download_mojo(best_xgb, path=paste0("~/filepath/",f))
  
  #Function to remove excess modelmetrics objects on h2o cluster
  h2oRemoveModelMetrics <- function() {
    df <- h2o.ls()
    keys_to_remove <- grep("^modelmetrics.*", perl=TRUE, x=df$key, value=TRUE)
    unique_keys_to_remove = unique(keys_to_remove)
    if (length(unique_keys_to_remove) > 0) {
      h2o.rm(unique_keys_to_remove)
    }
  }
  
  #Function to alculate performance on test set
  perf <- function(data,indices) {
    df = data[indices,]
    hdf = as.h2o(df)
    p = h2o.performance(best_xgb,hdf)
    auc = round(h2o.auc(p),3)
    mse = round(h2o.mse(p),3)
    aucpr = round(h2o.aucpr(p),3)
    f1 = h2o.F1(p) %>% arrange(-f1) %>% filter(row_number() == 1)
    precision = h2o.precision(p) %>% filter(threshold == f1$threshold[1])
    recall = h2o.recall(p) %>% filter(threshold == f1$threshold[1])
    output = c(auc, mse, aucpr, round(f1$f1,3), round(precision$precision,3), round(recall$tpr,3))
    names(output) = c("auc", "mse", "aucpr", "f1", "precision", "recall")
    
    #Clean up h2o cluster to save memory
    h2o.rm(h2o.getId(hdf))
    h2oRemoveModelMetrics()
    
    return(output)
  }
  
  #Overall metrics
  metrics = perf(test)
  
  #Bootstrapped performance
  results = boot(data=test, statistic=perf, R=1000)
  
  #Confidence intervals- percentiles from bootstrap samples
  aucci = tail(boot.ci(results,type="perc",index=1)$percent[1,],2)
  mseci = tail(boot.ci(results,type="perc",index=2)$percent[1,],2)
  aucprci = tail(boot.ci(results,type="perc",index=3)$percent[1,],2)
  f1ci = tail(boot.ci(results,type="perc",index=4)$percent[1,],2)
  precisionci = tail(boot.ci(results,type="perc",index=5)$percent[1,],2)
  recallci = tail(boot.ci(results,type="perc",index=6)$percent[1,],2)
  
  #Output- "Estimate (LCL, UCL)"
  metrics["auc"] = paste0(metrics[["auc"]]," (",paste(round(aucci, 3), collapse = ", "),")")
  metrics["mse"] = paste0(metrics[["mse"]]," (",paste(round(mseci, 3), collapse = ", "),")")
  metrics["aucpr"] = paste0(metrics[["aucpr"]]," (",paste(round(aucprci, 3), collapse = ", "),")")
  metrics["f1"] = paste0(metrics[["f1"]]," (",paste(round(f1ci, 3), collapse = ", "),")")
  metrics["precision"] = paste0(metrics[["precision"]]," (",paste(round(precisionci, 3), collapse = ", "),")")
  metrics["recall"] = paste0(metrics[["recall"]]," (",paste(round(recallci, 3), collapse = ", "),")")
  
  name = names(oclist[index])
  output = c(name,metrics)
  print(output)
  
  #Relative variable importance
  
  test_h2o = as.h2o(test)
  varimp <- h2o.varimp(best_xgb,test_h2o)
  write.csv(varimp,paste0('~/filepath/',f,'/VarImp.csv'))
  
  #Mean SHAP values
  c <- h2o.predict_contributions(best_xgb,test_h2o)
  means <- h2o.mean(abs(c),axis=0,return_frame=T)
  result <- data.frame(variable = names(c), mean_shap = as.vector(means)) %>% arrange(-mean_shap)
  write.csv(result,paste0('~/filepath/',f,'/SHAP.csv'))

  h2o.rm(h2o.getId(test_h2o))
  
  return(output)
}

#Main function
models_byoutcome <- function (lname,train,test,folder_prefix) {
  
  #Run models for each outcome sequentially
  mods = list()
  for (i in seq_along(lname)) {
    mods[i] = seq_out(index=i,
                      oclist=lname,
                      train=train,
                      test=test,
                      prefix=folder_prefix)
  }
  
  #assemble the results and export
  output = as.data.frame(do.call(rbind,mods))
  print(output)
  #colnames(output) = c("model","auc", "mse", "aucpr", "f1", "precision", "recall")
  write.csv(output,paste0("~/filepath/",folder_prefix,"_results.csv"))
  
  return(output)
}


results = models_byoutcome(outcomes,ref_train,ref_test,"ref")
results = models_byoutcome(outcomes,mref_train,mref_test,"mref")
results = models_byoutcome(outcomes,pef_train,pef_test,"pef")
results = models_byoutcome(outcomes,all_train,all_test,"all")

h2o.removeAll()
h2o.shutdown(prompt=F)

