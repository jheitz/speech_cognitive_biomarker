
library(psych)
library(semTools)
library(semPlot)
library(paran)
library(lavaan)
library(psych)
library(ggplot2)
library(GPArotation)
library(dplyr)
library(ltm)
library(corrplot)



setwd("/Users/jheitz/git/luha-prolific-study/src/factor_analysis")



load_and_rescale_data <- function(data_split_path) {
  data <- merge(
    read.csv(paste("/Users/jheitz/git/luha-prolific-study/data/processed_combined/", data_split_path, sep="")),
    read.csv("/Users/jheitz/git/luha-prolific-study/data/processed_combined/data/language_task_scores.csv"),
    by = "study_submission_id"
  )
  print("Dimension of data")
  print(dim(data))
  
  # Invert specified columns (higher = better)
  cols_to_invert <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                      "avg_reaction_speed", "place_the_beads_total_extra_moves", 
                      "fill_the_grid_total_time", 
                      "typeskill_time", "clickskill_time", "dragskill_time")
  data[cols_to_invert] <- data[cols_to_invert] * -1
  
  # Rescale specified columns (to make variances more similar)
  cols_to_rescale <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                       "fill_the_grid_total_time", "typeskill_time", 
                       "clickskill_time", "dragskill_time", "avg_reaction_speed")
  cols_to_rescale_factor <- c(1000, 1000, 1000, 1000, 1000, 1000, 10)
  
  for (i in seq_along(cols_to_rescale)) {
    data[[cols_to_rescale[i]]] <- data[[cols_to_rescale[i]]] / cols_to_rescale_factor[i]
  }
  
  study_submission_ids = data$study_submission_id
  
  # Select the specified columns
  cols_to_keep <- c('connect_the_dots_I_time_msec',
                    'connect_the_dots_II_time_msec', 'wordlist_correct_words'
                    ,'avg_reaction_speed'
                    , 'place_the_beads_total_extra_moves'
                    ,'box_tapping_total_correct', 'fill_the_grid_total_time'
                    ,'wordlist_delayed_correct_words', 'wordlist_recognition_correct_words'
                    ,'digit_sequence_1_correct_series', 'digit_sequence_2_correct_series'
                    #,'typeskill_levenstein_distance', 'wordlist_learning'
                    #,'typeskill_time'
                    , 'clickskill_time', 'dragskill_time'
                    ,"semantic_fluency_score","phonemic_fluency_score",  "picture_naming_score"
                    #, 'digit_sequence_difference'
                    , "study_submission_id"
  )
  data <- data[cols_to_keep]
  
  # Keep only complete rows
  data_complete_rows <- data[complete.cases(data), ]
  
  study_submission_ids_complete_rows = data_complete_rows$study_submission_id
  
  data <- subset(data, select = -c(study_submission_id))
  data_complete_rows <- subset(data_complete_rows, select = -c(study_submission_id))
  
  return(list(data = data, data_complete_rows = data_complete_rows, study_submission_ids = study_submission_ids, study_submission_ids_complete_rows = study_submission_ids_complete_rows))
}

data_split1 <- load_and_rescale_data("split1/acs_outcomes_outliers_removed.csv")
split1 <- data_split1$data
split1_complete_rows <- data_split1$data_complete_rows

#data_path <- "split2/acs_outcomes_outliers_removed.csv"
data_split2 <- load_and_rescale_data("split2/acs_outcomes_outliers_removed.csv")
split2 <- data_split2$data
split2_complete_rows <- data_split2$data_complete_rows

#data_path <- "data/acs_outcomes_imputed.csv"
data_all_imputed_list <- load_and_rescale_data("data/acs_outcomes_imputed.csv")
data_all_imputed <- data_all_imputed_list$data
data_all_imputed_complete_rows <- data_all_imputed_list$data_complete_rows



names(split1)




## Check standard deviation of variables
# str(EFA) # checks kind of variable (int, num) and values
sapply(split1, sd, na.rm = TRUE)
# question: SD is unclear, depends on magnitude of variable?

## Which factors are related to each other? Could look for initial ZHe
# note: complete.obs = missing values (=outliers) are handled byp casewise deletion
cor_matrix = cor(split1_complete_rows, use='complete.obs')
corrplot(cor_matrix)

### Requirements
## Correlation matrix
# not correct for EFA: cov_matrix <- cov(EFA)
# note: complete.obs = missing values (=outliers) are handled by casewise deletion
cor_matrix <- cor(split1_complete_rows, use='complete.obs')


# get highest correlations
cor_matrix_copy <- cor_matrix
diag(cor_matrix_copy) <- 0
max_correlations <- apply(cor_matrix_copy, 1, function(x) max(abs(x)))
ordered_variables <- max_correlations[order(-max_correlations)]
print(ordered_variables)


# VIF
model <- lm(wordlist_correct_words ~ wordlist_delayed_correct_words + wordlist_recognition_correct_words, data = split1_complete_rows) #
#vif(model)



## KMO: Kaiser-Meyer-Olkin factor adequacy
### measure of sampling adequacy that "indicates the proportion of variance in […] variables that might be caused by underlying factors"
### In general, KMO values between 0.8 and 1 indicate the sampling is adequate. KMO values less than 0.6 indicate the sampling is not adequate and that remedial action should be taken
kmo_result <- KMO(cor_matrix)
print(kmo_result)
# overall msa = 0.73 (-> middling sampling adequacy)

# inspect anti-image covariance matrix -> should approach diagonal matrrix
corrplot(cor_matrix)
corrplot(kmo_result$Image)
#corrplot(kmo_result$ImCov)

## Sphericity (Bartlett's)
### Bartlett’s Test of Sphericity assesses whether the correlation matrix built before is an identity matrix, which would indicate that the variables are unrelated and thus unsuitable for factor analysis
bartlett_test_result <- cortest.bartlett(cor_matrix, n=nrow(split1_complete_rows))
print(bartlett_test_result)
# Chi2 = 1090.654 (sign. diff. from identity matr.), p-value = 2.461314e-192, df = 55; indicates that data is highly intercorrelated (not independent); high df bcs sooo many variables

### Factor number
## Choose how many factors to retain -> choose extraction (estimation) method
## Eigenvalue
EV <- eigen(cor_matrix)$value
EV
# 4 > 1
## Elbow
EV_dat <- data.frame("Eigen_Values"= EV,"Index" = 1:length(EV))
ggplot(data = EV_dat) +
  geom_point(aes(x = Index,y = Eigen_Values)) +
  geom_line(aes(x = Index,y = Eigen_Values)) +
  geom_hline(yintercept = 1, lty = 2) +
  theme_minimal()


## Velicer's Minimum Average Partial (MAP) Test
VSS(cor_matrix, n=8, n.obs = nrow(split1_complete_rows), rotate='oblimin')
# VSS implies 2 factors better than 1
# MAP implies 8 factors; 8f achieves minimum average partial corr; BIC (penalizes when a lot of variables) also recommends 8...
# statistics imply 8 factors, but theoretical implications and usability may indicate less factors, albeit 2 factors would probably too little
## Parallel Analysis
fa1 <- fa.parallel(cor_matrix,fa="fa", n.obs = nrow(split1_complete_rows))
# abline(h=mean(fa1$fa.values),lty=2)
# or: fa1 <- fa.parallel(EFA,fa="fa")
# 14




# Specify the CFA model
model <- '
  memory =~ wordlist_correct_words +  wordlist_delayed_correct_words + wordlist_recognition_correct_words
  language =~ semantic_fluency_score + phonemic_fluency_score + picture_naming_score
  speed =~ avg_reaction_speed + fill_the_grid_total_time + clickskill_time + dragskill_time 
  executive_function =~ connect_the_dots_I_time_msec + connect_the_dots_II_time_msec + digit_sequence_1_correct_series + digit_sequence_2_correct_series 
  digit_sequence_1_correct_series ~~    digit_sequence_2_correct_series
  speed =~       connect_the_dots_I_time_msec
'


# Fit the CFA model
fit <- cfa(model = model, data = split2_complete_rows, rotation = "oblimin", std.lv = TRUE, estimator = "MLR")


fit <- cfa(model = modelE6C, data = data_all_imputed_complete_rows, std.lv = TRUE, estimator = "MLR")

# Summarize the results
fitMeasures(fit, c("cfi.robust", "tli.robust", "rmsea.robust", "srmr", "cfi.scaled", "cfi"))
options(max.print = 2000)
summary(fit, fit.measures = TRUE, standardized = TRUE, modindices = TRUE, estimates=TRUE, ci=TRUE)

residuals(fit, type="standardized")
residuals(fit, type="cor")

# Visualize the CFA model
png("cfa_model.png", width = 2000, height = 1400, res = 200)  # tweak as needed
semPaths(fit, "std", layout = "tree", style = "lisrel", residuals = TRUE, rotation=2,
         whatLabels = "std", edge.label.cex = 1, nCharNodes = 30,
         nDigits = 2, sizeMan = 15, sizeLat = 10, sizeMan2 = 5)
dev.off()



factorScores <- data.frame(lavPredict(fit, data_all_imputed_complete_rows))
factorScores$study_submission_id <- data_all_imputed_list$study_submission_ids_complete_rows
factorScores <- merge(
  data.frame(study_submission_id = data_all_imputed_list$study_submission_ids),
  factorScores,
  all.x = TRUE
)


output_filename <- paste("/Users/jheitz/git/luha-prolific-study/src/resources/factor_scores_theory_", format(Sys.time(),"%Y-%m-%d-%H%M"), ".csv", sep="")
print(output_filename)
write.csv(factorScores,output_filename, row.names = FALSE)






