library(relaimpo)
# Age-> 1:Old 0:Young
# Gender-> 1:Male 0:Female
# Race -> 1:African American/Black 0:European American/White
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
args = commandArgs(trailingOnly=TRUE)
data_dir = './predictions'
fname = 'reg_analysis_input_race_gender_age.csv'
#### MODELS ### 
#model_name='cnn' 
#model_name='rnn'
#model_name='attn'    
#model_name='svm'
#model_name='bert'

model_name <- args[1] # it should be one of the followings: ['cnn','rnn','attn',svm']
model = paste(data_dir,model_name,fname,sep='/')
df_race_gender = read.csv(model,header = FALSE,col.names=c("Name", "Intensity", "Vector_Magnitude", "Age", "Gender", "Race"))
# Remove Duplicates (each entity written twice because race/gender uses same set of names)
df_race_gender = unique(df_race_gender)
# Scale Vector Magnitudes 
df_race_gender[c(3)] <- lapply(df_race_gender[c(3)], function(x) c(range01(x)))

ols.sat<-lm(Intensity~ Race + Gender + Vector_Magnitude + Age,data=df_race_gender)
summary(ols.sat)
#lm.beta(ols.sat)
calc.relimp(ols.sat, type = c("lmg"), rela = FALSE)

