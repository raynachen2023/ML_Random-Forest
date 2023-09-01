# sound-of-music-
## Data Analysis Process 


Based on the understanding of the data, at the very first few attempts, I aggregated categories for some variables and divided numbers into groups. For example, 

Conveyed track_duration to track_mins;

Categorized danceability into two categories;

Tried to find some patterns for variable ‘performer’, for example, see whether the performer with ‘the’ in it has significantly different ratings;

Found ‘easy listening’ patterns for variable ‘Genre’;

Got rid of negative ratings as it could be a typo or simply people did not rate

Then I used linear regression to run the model.  The RMSE was around 16.3 which is pretty high. Then I dived more deep into other variables, especially ‘genre’ which I believe there could be a lot of information to get. 

I looked at ‘pop’, ‘funk’, ‘rock’, ‘rap’, ‘soul’, ‘hiphop’, ‘country’, ‘standards’ , ‘classic’, ‘blues’ , ‘album’. It turned out that there were no significant differences for class, blues and album on ratings. If the genre falls under ‘pop’, ‘funk’, ‘rock’, ‘rap’, ‘soul’, ‘hiphop’, ‘country’, ‘standards’  people are likely to rate significantly differently. Below are the results from linear regression.  It is also interesting to see that for soul and hip hop songs, people tend to give lower ratings compared to other types of music. Moreover, if it is rap music, people tend to rate about 5.5 points higher. This helps us understand what types of music people love to listen to. Below are the results: 
variable, sd, t-value, p value

loudness                   0.78400    0.04692  16.710  < 2e-16 ***

pop_status                 3.92524    0.23249  16.883  < 2e-16 ***

funk_status                2.25112    0.45784   4.917 8.86e-07 ***

rap_status                 5.50642    0.51749  10.641  < 2e-16 ***

rock_status                3.09047    0.25059  12.333  < 2e-16 ***

soul_status               -2.67415    0.36368  -7.353 2.01e-13 ***

hiphop_status             -3.36995    0.52584  -6.409 1.50e-10 ***
 
After I ran a couple of linear regressions, the RMSEs I got went down to around 15:8.  Then I switched my strategies to implementing other statistical methods. I tried out the decision tree using rpart with the same variables. However, it turned out that the RMSE was even higher. It could be due to overfitting and unstable of the trees.  

After examinations, I found out some of the best variables that significantly affects rating, which are: acousticness + tempo + folk_status + mode+ instrumentalness+ liveness + time_signature + danceability_one_half  +  track_mins + track_explicit + energy + loudness + pop_status+ funk_status + rap_status + rock_status + soul_status + hiphop_status + valence.


## My Best Model 
After a few tries, the best model of mine is using a tuned  random forest method. In order to avoid overfitting, I implemented 5-fold cross-validation, and set the max mtry to be the number of columns minus 1. I also set the number of trees to be 200 because I tried 1000 first but the model ran extremely slow (it did not give me the result after 20mins). Therefore, I set the number of trees a bit less.  The public and private board showed pretty similar results. For the training set it was about 15.19, the public testing set was about 15.38 and the private board was about 15.2. Even though there was still some room to improve RMSE, it shows that there was not much variances and the model is not overfitting. 

Some of the variables I kept the way it is, for example, acousticness, tempo, mode,instrumentalness, liveness, track_explicit, energy, loudness,valence. Then, I picked genres that had a larger effect on ratings. Random forest is a great model here as it can be used for both classification and regressions.

## Failed Steps or Missteps - Something Could be Improved
Here are some missteps when I explored numeric variables.  I jumped quickly  into categorizing  them by quartiles, for example, dividing loudness into 4 groups based on 4 quartiles. I did that for other numeric variables as well. However, after utilizing linear regressions to see the correlations I found out that their correlations were not significant compared to non-transformation. Then, I switched my strategies to keep some of the numeric variables the way they were. However, I believe it was a good try. There might be certain cases that need to transform numeric data into categories, for example, age groups. 

Also, for my last few attempts, I tried boosting strategies with cross-validation and xgboost. However, when using gbm for boosting, the computer ran extremely slow and did not give results after half an hour and then I stopped attempting using the method. Therefore, I was not able to get results from running gbm. 

I also tried Xgboost, however, it worked too well for the training data, which means it was extremely overfitting. I received less than 1 RMSE for the first try and when moving to testing data the RMSE became about 15.7. Even after I changed the number of rounds and some other parameters, it kept overfitting. What could have been improved is to do appropriate tuning. It may help change the overfitting. 

Another thing I could have improved on is to make every genre as a binary variable. There are over a thousand types of genres in the data set. In my past model, I only picked a few common genres for analysis. It could have given me better results if I listed out every type of genre to run the model. 

Overall, this Kaggle competition is a great way for me to apply data tidying and different statistical methods to solving actual problems. After trying different methods I was able to get familiar with the process of data analysis, for example how to start with a data analysis, and what should be done first (data tidying and partition) and what methods can be used, which are some of the key techniques in problem solving. 




**Best submission: Load data**

```{r}
library(dplyr)
library(ggplot2)
library(randomForest)
library(lattice)
library(caret)
dat <- read.csv('C:/Users/chenron/Desktop/Columbia/R/Kaggle/Competition Data/analysisData.csv')
dim(dat)
test<- read.csv('C:/Users/chenron/Desktop/Columbia/R/Kaggle/Competition Data/scoringData.csv')
dim(test)

```




**Data exploration & tidying**
```{r}
#convert to mins for training set
dat <- dat %>% 
  mutate(track_mins = dat$track_duration/(1000*60)) 
test <- test %>% 
  mutate(track_mins = test$track_duration/(1000*60)) 

#categorize danceability 
dat <- dat %>% 
  mutate(danceability_one_half = (danceability>=0.5)) 

test <- test %>% 
  mutate(danceability_one_half = (danceability>=0.5)) 

#shift the negative ratings to zero ratings. -1 may represents someone did not rate 
dat$rating <- pmax(0, dat$rating) 

#aggregate the categories in some way - time_signiature  3, 4 categories, and other 
dat$time_sig_cat <- character(length = nrow(dat))
dat$time_sig_cat[dat$time_signature == 4] <- "Four"
dat$time_sig_cat[dat$time_signature == 3] <- "Three"
dat$time_sig_cat[dat$time_signature %in% c(0,1,5)] <- "Others"

test$time_sig_cat <- character(length = nrow(test))
test$time_sig_cat[test$time_signature == 4] <- "Four"
test$time_sig_cat[test$time_signature == 3] <- "Three"
test$time_sig_cat[test$time_signature %in% c(0,1,5)] <- "Others"

#classify genre 
pop <- grep(pattern = 'pop', x = dat$genre)
dat$pop_status <- 0
dat$pop_status[pop] <-1


funk <- grep(pattern = 'funk', x = dat$genre)
dat$funk_status <- 0
dat$funk_status[funk] <-1


rock <- grep(pattern = 'rock', x = dat$genre)
dat$rock_status <- 0
dat$rock_status[rock] <-1


rap <- grep(pattern = 'rap', x = dat$genre)
dat$rap_status <- 0
dat$rap_status[rap] <-1

soul <- grep(pattern = 'soul', x = dat$genre)
dat$soul_status <- 0
dat$soul_status[soul] <-1


hiphop <- grep(pattern = 'hip hop', x = dat$genre)
dat$hiphop_status <- 0
dat$hiphop_status[hiphop] <-1


standards <- grep(pattern = 'standards', x = dat$genre)
dat$standards_status <- 0
dat$standards_status[standards] <-1


country <- grep(pattern = "\'country\'", x = dat$genre)
dat$country_status <- 0
dat$country_status[country] <-1

folk <- grep(pattern = 'folk', x = dat$genre)
dat$folk_status <- 0
dat$folk_status[folk] <-1


adult <- grep(pattern = 'adult standards', x = dat$genre)
dat$adult_status <- 0
dat$adult_status[adult] <-1


#classify genre in testing set
pop <- grep(pattern = 'pop', x = test$genre)
test$pop_status <- 0
test$pop_status[pop] <-1


funk <- grep(pattern = 'funk', x = test$genre)
test$funk_status <- 0
test$funk_status[funk] <-1


rock <- grep(pattern = 'rock', x = test$genre)
test$rock_status <- 0
test$rock_status[rock] <-1


rap <- grep(pattern = 'rap', x = test$genre)
test$rap_status <- 0
test$rap_status[rap] <-1


soul <- grep(pattern = 'soul', x = test$genre)
test$soul_status <- 0
test$soul_status[soul] <-1


hiphop <- grep(pattern = 'hip hop', x = test$genre)
test$hiphop_status <- 0
test$hiphop_status[hiphop] <-1


standards <- grep(pattern = 'standards', x = test$genre)
test$standards_status <- 0
test$standards_status[standards] <-1


country <- grep(pattern = 'country', x = test$genre)
test$country_status <- 0
test$country_status[country] <-1


folk <- grep(pattern = 'folk', x = test$genre)
test$folk_status <- 0
test$folk_status[folk] <-1



```

**Using random forest & make predictions**
```{r}
set.seed(1031)
trControl = trainControl(method = 'cv', number = 5)
tuneGrid = expand.grid(mtry = 1:ncol(dat)-1)
rf1 <- randomForest(rating ~ acousticness + tempo + folk_status + mode+ instrumentalness+ liveness + time_signature + danceability_one_half  +  track_mins + track_explicit + energy + loudness + pop_status+ funk_status + rap_status + rock_status + soul_status + hiphop_status + valence,data = dat, ntree=200, norm.votes=FALSE)

pred_train = predict(rf1)
rmse_train_forest = sqrt(mean((pred_train - dat$rating)^2)); rmse_train_forest  
