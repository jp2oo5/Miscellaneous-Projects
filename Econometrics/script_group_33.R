#Libraries------------------------------------------------
library(tidyverse)
library(lmtest)
library(car)
library(tidyr)
library(ggplot2)
library(ggcorrplot)

# Importing and Preprocessing the Dataset----------------------

ball <-read.csv("NCAA D1 Bball Data 23-24.csv")
#Creating and removing columns
ball <- ball %>%
  mutate(DRB=TRB-ORB, X2P= FG- X3P, X2PA= FGA- X3PA, X2PF=X2PA- X2P, X3PF=X3PA- X3P, FTF= FTA-FT)%>%
  subset(select=-c(Rk, X, X.1, X.2, X.3, X.4, W, L ,SRS, ConfW, ConfL, HomeW, HomeL, AwayW, AwayL, PointsTm, PointsOpp, MP, FG_pct, X3P_pct, FT_pct,TRB, FG, FGA, X2PA, X3PA, FTA))%>%
  rename(Win_ratio=W.L_pct)
#Finding NA
which(is.na(ball), arr.ind=TRUE)
#Removing NA and ordering columns 
ball <- ball[1:362, c(1,2,3,4,14,15,5,16,6, 17,13, 7,8,9,10,11,12)]

#Checking for possible log transformations
vars<- c("Games","SOS","X2P","X2PF","X3P","X3PF","FT","FTF","ORB","DRB","AST","STL","BLK","TOV","PF")

for (var in vars) {
  plot(ball$Win_ratio, ball[[var]], xlab="Win Ratio", ylab = var)
}
#No possible log transformation found 

#-------------------------Model 1--------------------------------------
#Simple regression with all the variables
attach(ball)
reg<-lm(Win_ratio~Games+ SOS+ X2P+ X2PF +X3P+ X3PF+ FT+ FTF+ ORB+ DRB+ AST+ STL+ BLK+ TOV+ PF)
summary(reg)
#High R-Squared and Adjusted R-Squared

# MLR Assumptions---------------------------------------------------

# Linear in parameters
#Plot of the residuals
plot(reg, which = 1)
#Does not seem to be any non linear relation between the dependent and independent variables,
#as the red line is relatively flat, we only have problems when the fitted values
#approach 1, which is fair, as we don't want a win ratio above 1, that is impossible
#This might be one weakness in our model, that it may predict win ratios above 1

# Random Sampling
#We have a random sample of 362 observations from one basketball season.

# No Perfect Collinearity
correl<- cor(ball[,vars], , use = "complete.obs")
ggcorrplot(correl, 
           method = "square",          
           type = "lower",           
           lab = TRUE ,
           lab_size = 2)
#There are no perfectly correlated variables
#However, we must be careful with X3P and X3PF, as they have a correlation of 0.85

# Zero Conditional Mean
res<-resid(reg)
#Mean of residuals
mean(res)
#The mean of the residuals is very close to zero, assuring us of the zero conditional mean

# Homoscedasticity
#BP Test
bptest(reg)
#Special White
bptest(reg, ~ fitted(reg) + I(fitted(reg)^2) )
#Breusch-Pagan and Special White Tests 
#Don't reject the null hypothesis, no sign of heteroskedasticity
#We don't need to use the robust estimators

# Normality
#Histogram residuals
hist(res)
#Resembles a normal distribution

# RESET Test-------------------------------------------
reset <- lm(Win_ratio~Games+ SOS+ X2P+ X2PF +X3P+ X3PF+ FT+ FTF+ ORB+ DRB+ AST+ STL+ BLK+ TOV+ PF+ I(fitted(reg)^2) + I(fitted(reg)^3))
resettest(reset)
# We don't reject H0, there is no evidence of model misspecification

# ------------------------------Model 2----------------------------------
#Using only coefficients at the 5% significance level
#We also remove one of the two highly correlated coefficients, strengthening the
#no perfect collinearity assumption
reg2<- lm(Win_ratio~Games+ SOS+ X2P+ X2PF + X3PF+ FT+ FTF+ ORB+ DRB+ AST+ STL+ TOV)
summary(reg2)
#High R-Squared and Adjusted R-Squared again

# MLR Assumptions----------------------
# MLR 2 and 3 can be reused from the previous model

#Linear in parameters
plot(reg2, which = 1)
#Same problem as before, line deviates as the fitted values approach 1

#Zero Conditional Mean
res2<-resid(reg2)
mean(res2)
#The mean of the residuals is very close to zero once again

# Homoscedasticity
#BP Test
bptest(reg2)
#Special White
bptest(reg2, ~ fitted(reg2) + I(fitted(reg2)^2) )
#Once again, no sign of heteroskedasticity, we don't need to use the robust estimators

#Normality
#Histogram residuals
hist(res2)
#Resembles a normal distribution

# RESET Test-------------------------------------------------
reset2 <- lm(Win_ratio~Games+ SOS+ X2P+ X2PF + X3PF+ FT+ FTF+ ORB+ DRB+ AST+ STL+ TOV+ I(fitted(reg2)^2) + I(fitted(reg2)^3))
resettest(reset2)
#Once again, no model misspefication evidence

#-----------------------------Model 3-------------------------------------
#Literature says that shooting percentages and defensive rebounds are the most 
#important aspects
reg3<- lm(Win_ratio~DRB+ I(X2P/(X2PF+X2P))+I(X3P/(X3PF+X3P))+I(FT/(FTF+FT)))
summary(reg3)
#Lower R-Squared and Adjusted R-Squared compared to before

#Linear in parameters
plot(reg3, which = 1)
#Still the same problem

#Zero Conditional Mean
res3<-resid(reg3)
mean(res3)
#The mean of the residuals is very close to zero once again

# Homoscedasticity
#BP Test
bptest(reg3)
#Special White
bptest(reg3, ~ fitted(reg3) + I(fitted(reg3)^2) )
#Once again, no sign of heteroskedasticity, we don't need to use the robust estimators

# Normality
#Histogram residuals
hist(res3)
#Resembles a normal distribution again

#RESET Test
reset3 <- lm(Win_ratio~DRB+ I(X2P/(X2PF+X2P))+I(X3P/(X3PF+X3P))+I(FT/(FTF+FT))+ I(fitted(reg3)^2) + I(fitted(reg3)^3))
resettest(reset3)
#Once again, no model misspefication evidence

###################### Final Model################################
#All the models had good MLR assumptions and passed the RESET test
#Choose the second model
#Close R-Squared and Adjusted R-Squared to the first model
#But removed the correlated variable and variables not significant at the 5% level
#Let's predict some win ratios from the this season, 2024-2025

#Dataset from 2024-2025
ball2<-read.csv("NCAA D1 Bball Data 24-25.csv")
#Creating and removing columns
ball2 <- ball2 %>%
  mutate(DRB=TRB-ORB, X2P= FG- X3P, X2PA= FGA- X3PA, X2PF=X2PA- X2P, X3PF=X3PA- X3P, FTF= FTA-FT)%>%
  subset(select=-c(Rk, X, X.1, X.2, X.3, X.4, W, L ,SRS, ConfW, ConfL, HomeW, HomeL, AwayW, AwayL, PointsTm, PointsOpp, MP, FG_pct, X3P_pct, FT_pct,TRB, FG, FGA, X2PA, X3PA, FTA))%>%
  rename(Win_ratio=W.L_pct)
#Finding NA
which(is.na(ball), arr.ind=TRUE)

#Predicting
pred<-as.numeric(predict(reg2, ball2))
df<- ball2%>%
  subset(select=c(School, Win_ratio))%>%
  mutate(pred_win_ratio=pred, diff=pred_win_ratio-Win_ratio)
#Plotting our prediction against the actual values
plot(df$Win_ratio, df$pred_win_ratio, xlab = "Win Ratio", ylab="Predicted Win Ratio")
abline(0,1)

#It looks pretty good
