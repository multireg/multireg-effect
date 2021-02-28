library(pscl)
library(dominanceanalysis)
library(DescTools)
set.seed(34)
args = commandArgs(trailingOnly=TRUE)
# Variables
#### COREFERENCE RESOLUTION MODELS ####
#model_name = "lee13.tsv"
#model_name = "lee17.tsv"
#model_name = "clark_and_manning.tsv"
#model_name = "wiseman.tsv"
#model_name = "lee18.tsv"
#model_name = "spanbert_large.tsv"
mname = args[1]
model_name = paste(mname,'tsv',sep='.')
model = paste('gap_predictions',model_name,sep='_')
prediction_path = "predictions"

extra_variables = paste("extra_variables_gap_development",model_name,sep='_')
extra_variables = paste(prediction_path,extra_variables,sep='/')
prediction_file = paste(prediction_path,model,sep='/')

# Read predictions
preds <- read.table(prediction_file,sep='\t',header=FALSE)
colnames(preds) <- c("ID", "A_coref","B_coref")
# Read extra input variables file
extra_vars <- read.table(extra_variables,sep='\t',header=TRUE)

# Ignore instances in prediction file where gold predictions are FALSE,FALSE
preds_subset = preds[preds$ID %in% extra_vars$ID,]

# Merge extra_vars and preds subsets.
merged = merge(preds_subset,extra_vars,sort=F)

#Normalize & preprocess extra variable colums.
# Add small constant and take log of frequencies
constant = 1e-20
merged$correct_freq   <- merged$correct_freq + constant
merged$incorrect_freq <- merged$incorrect_freq + constant
merged$correct_freq   <- log(merged$correct_freq,2)
merged$incorrect_freq <- log(merged$incorrect_freq,2)
max_val = max(abs(min(merged[,c('correct_diff','incorrect_diff')])),abs(max(merged[,c('correct_diff','incorrect_diff')])))
merged$correct_diff <- merged$correct_diff / max_val
merged$incorrect_diff <- merged$incorrect_diff / max_val

# Convert TRUE/FALSE in A_coref,B_coref,A_coref_gold,B_coref_gold,option_1_label,option_2_label columns to 1/0
merged$A_coref <- as.numeric(merged$A_coref) - 1
merged$B_coref <- as.numeric(merged$B_coref) - 1
merged$A_coref_gold <- as.numeric(merged$A_coref_gold) - 1
merged$B_coref_gold <- as.numeric(merged$B_coref_gold) - 1
merged$option_1_label <- as.numeric(merged$option_1_label) - 1
merged$option_2_label <- as.numeric(merged$option_2_label) - 1
# Convert Male/Female to 0/1
merged$gender <- ifelse(merged$gender=="female",1,0)

# Perform logistic regression.
# Define logistic regression label for each instance
# Option#1 : Only check if the True candidate was detected correctly.
# Option#2 : Check if both of the candidates were detected correctly. [USE THIS to replicate the experiments in the paper]

shuffled_merged  <- merged[sample(nrow(merged)),]
folds <- cut(seq(1,nrow(shuffled_merged)),breaks=10,labels=FALSE)
model_accs <- c()
baseline_accs <- c()
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test <- merged[testIndexes, ]
  train <- merged[-testIndexes, ]
  model <- glm(option_2_label ~ gender +correct_freq + incorrect_freq + correct_diff + incorrect_diff + correct_single + incorrect_single + correct_same + incorrect_same,
               family=binomial(link='logit'),data=train) 
  model_prob = predict(model,test, type="response")
  model_pred = 1*(model_prob > .50) + 0
  gold_label = test$option_2_label
  model_acc  = sum(1*(model_pred == gold_label))/length(gold_label)
  most_freq_label = Mode(merged$option_2_label)
  baseline_acc   = sum(gold_label==most_freq_label)/length(gold_label)
  model_accs <- append(model_accs,model_acc)
  baseline_accs <- append(baseline_accs,baseline_acc)
}
mean(model_accs)
mean(baseline_accs)
model <- glm(option_2_label ~ gender +correct_freq + incorrect_freq + correct_diff + incorrect_diff + correct_single + incorrect_single + correct_same + incorrect_same,
             family=binomial(link='logit'),data=merged) 

pR2(model)
PseudoR2(model, c("McFadden", "Nagel","VeallZimmermann","McKelveyZavoina"))

# Likelihood Ratio Test
#anova(model, test ="Chisq")

# Dominance Analysis for Relative importance.
dapres<-dominanceAnalysis(model)
# Explore general dominance by:
averageContribution(dapres,fit.functions = "r2.m")
