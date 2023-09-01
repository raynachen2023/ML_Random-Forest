# sound-of-music-
Data source: https://www.kaggle.com/competitions/musiclala2023
The project is using random forest method to predict ratings

## Understand the data
id: Song id

performer: Performer name

song: Song name

genre: Genre

track_duration: Duration in milliseconds

track_explicit: Is explicit

danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a 

sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
time_signature: Time signature

rating: Rating derived from Billboard Top 100 and popularity on Spotify. The values of rating were constructed by me and are not meant to be used outside of the context of this class exercise.

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


## Model 
After a few tries, the best model of mine is using a tuned  random forest method. In order to avoid overfitting, I implemented 5-fold cross-validation, and set the max mtry to be the number of columns minus 1. 
Some of the variables I kept the way it is, for example, acousticness, tempo, mode,instrumentalness, liveness, track_explicit, energy, loudness,valence. Then, I picked genres that had a larger effect on ratings. Random forest is a great model here as it can be used for both classification and regressions.


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
