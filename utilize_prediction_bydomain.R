################################################
#
# @Organization - Kaiser Permanente Northern California, Division of Research
#
# @Project - Developing Clinical Risk Prediction Models for Worsening Heart Failure Events 
#            and Death by Left Ventricular Ejection Fraction 
#
# @Description - This program develops prediction models for worsening heart failure and death 
#                using variables from each EHR data domain in a forward selection procedure.
#
# @Output - This program will output a file for each outcome with the results of forward selection by
#           EHR data domain.
#
# @Data - The data referenced in this file is not publicly available, due to the inclusion of
#         Protected Health Information and risks of re-identification. Placeholder filepaths
#         are included in this program.    
################################################

#Load Packages
library(dplyr)
library(h2o)
library(parallel)
library(readr)

############ Data setup ############

#Initialize H2O cluster
h2o.init(nthreads=32,ip = "localhost",port = 54323)

#Import cvs to R environment
all_train = read_csv("~/filepath/all_train.csv")
all_test = read_csv("~/filepath/all_test.csv")

dfs <- list(all_train,all_test)
names(dfs) <- c("all_train","all_test")

#Clean data
#################

# Categorical variable list (including binary outcomes)- sex, race, tobacco use, alcohol use, drug use, 
# aortic stenosis severity, left ventricular hypertrophy, urine dipstick proteinuria, outcomes
catvars = c('demo_sex','demo_race','shx_tobacco_use','shx_alcohol_use','shx_illicit_drug_use','shx_iv_drug_use'
            ,'echo_as_severity_c','echo_lvh_c','lab_dip_c','whf_outcome','death_outcome','whf_outcome_ip'
            ,'whf_outcome_av','whf_outcome_edo')

#Convert categorical variables to factors
clean_df <- function (dsn) {
  dsn[catvars] = lapply(dsn[catvars],as.factor)
  return(dsn)
}

dfs_clean <- lapply(dfs,clean_df)

#Save as H2O frames
check <- lapply(dfs_clean,as.h2o)
list2env(check,globalenv())


### Modeling - Forward selection by EHR data domain ###
################

#Function for modeling
seq_dom <- function (index,ln,outcm,tr,te) {
  d <- unname(unlist(ln[index]))
  x1 <- c(x,d)
  modtm <- system.time ({
    
    #Xgboost with default settings
    mod <- h2o.xgboost(x = x1, y = outcm, training_frame = tr, nfolds=5,seed=1234,
                       ntrees = 1000,learn_rate=0.01, stopping_rounds = 5, stopping_tolerance = 2e-3, stopping_metric = "AUC",
                       sample_rate = 0.8,col_sample_rate = 0.8,score_tree_interval = 10
    )
    
  })
  
  auc <- h2o.auc(h2o.performance(mod,te))
  mse <- h2o.mse(h2o.performance(mod,te))
  time <- round(modtm[[3]],0)
  name <- names(ln[index])
  output <- list(c(name,auc,mse,time),mod)
  return(output)
}


#Function to loop through each data domain for a given outcome

loop_bydomain <- function (outcome) {
  
  #Initialize variables and groups
  ###
  
  #Identify subgroup domains- list to forward select over
  colnames <- colnames(all_train)
  
  #Demographic variables as baseline model
  demo <- colnames[grepl("^demo",colnames)]
  
  #Identify EHR data domains by variable prefix
  subset_domains <- function (lnames) {
    exp <- paste0("^",lnames)
    sub <- colnames[grepl(exp,colnames)]
    return(sub)
  }
  doms <- list('empty','census','shx','comorb','proc','lab','vital','rx','ecg','echo','nlp')
  dom_obj <- lapply(doms,subset_domains)
  names(dom_obj) <- doms
  
  #Output dataset
  fselection_final <- data.frame(matrix(ncol = 4, nrow = 0))
  names(fselection_final) <- c("model","auc","mse","time")
  
  #Initial predictors and outcomes
  x <- demo
  y <- outcome
  
  #Forward selection iteration counter
  iter <- 1
  
  
  ### Forward selection loop starts here
  repeat {
    
    #Run models for each set of variables in parallel
    mods <- mclapply(seq_along(dom_obj),seq_dom,
                     ln=dom_obj,
                     outcm=y,
                     tr=all_train,
                     te=all_test,
                     mc.cores=length(dom_obj))
    
    #extract the results
    results <- lapply(mods,function (x) x[[1]])
    output <- as.data.frame(do.call(rbind,results))
    colnames(output) <- c("model","auc","mse","time")
    
    #sort by auc and get top
    output <- output %>% arrange(desc(auc))
    top_lname <- output[1,1]
    
    #print selection step
    print(c("Iteration =",iter),quote=F)
    print(output)
    iter <- iter+1
    
    #Add top domain name to predictors vector
    b <- unlist(dom_obj[top_lname], use.names=F)
    x <- c(x,b)
    
    #Remove top domain from dom_obj list
    dom_obj <- dom_obj[names(dom_obj) != top_lname]
    
    #Append top model results to final table
    fselection_final <- rbind(fselection_final,output[1,])
    
    if(length(dom_obj)==0) {
      break
    }
  }
  
  print(fselection_final)
  write.csv(fselection_final,paste0("~/filepath/",outcome,"_fselect_bydom_results.csv"))
  
  return(fselection_final)
}

whf_output <- loop_bydomain("whf_outcome")
death_output <- loop_bydomain("death_outcome")


h2o.removeAll()
h2o.shutdown(prompt=F)










