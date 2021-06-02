# 1.a. Read in the data set
library(tree)
library(ISLR)
csv_download_link = "https://guides.newman.baruch.cuny.edu/ld.php?content_id=39910549"
link_url = url(csv_download_link) # Convert the text to URL format
data = read.csv(link_url)
str(data)

# 1.b Randomly split 80% of the data into the training set 
# and the remainder 20% into the test set. Use set.seed(1) 
# so that I can replicate your results.
set.seed(1) 
train.index = sample(1:nrow(data),nrow(data)*0.80)

# 1.c.
library(tree)
train = data[train.index,]
test = data[-train.index,]

summary(test)
summary(train)



#2a
set.seed(1)
train.index=sample(1:nrow(data),nrow(data)*0.80)
train=data[train.index,]
test=data[-train.index,]
model=tree(as.factor(y)~.,data = train)
summary(model)
plot(model)
text(model)

#2b
best.tree=cv.tree(model,K=10)
best.tree

x=best.tree$size
y=best.tree$dev
plot(x,y,xlab="Tree Size", ylab="Deviance",type="b",pch=20,col="blue")
model.pruned=prune.tree(model,best=4)

plot(model.pruned)
text(model.pruned)


#2c
model.pruned=prune.tree(model,best=4)
pred.class=predict(model.pruned,test,type="class")
c.matrix=table(test$y,pred.class);c.matrix

acc=mean(test$y==pred.class)
sens.high=c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec.high=c.matrix[1]/(c.matrix[1]+c.matrix[2])
data.frame(acc,sens.high,prec.high)

# 3a 1.	Check the min and max values for the continuous variables 
# to see if they are on even scale. 
# Normalize them to 0-1 scale if needed.	

summary(data)

library(caret)
preproc2 <- preProcess(data[,c(1:6,7)], method=c("range"))

norm2 <- predict(preproc2, data[,c(1:6,7)])
summary(norm2)

# Convert the categorical variables into dummy variables.
# Convert to character, so we can change values
normalize = function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

# Normalize balance and income
norm.age = normalize(data$age)
norm.balance = normalize(data$balance)
norm.duration = normalize(data$duration)
norm.campaign = normalize(data$campaign)
norm.data=cbind(data[,1:7],norm.age,norm.balance,norm.duration,norm.campaign)
summary(norm.data)
data$housing = as.character(data$housing)
data$housing[data$housing=="no"] = 0 
data$housing[data$housing=="yes"] = 1 
str(data$housing)
data$contact = as.character(data$contact)

data$contact[data$contact=="unknown"] = 0 
data$contact[data$contact=="cellular"] = 1 
data$contact[data$contact=="telephone"] = 1 
str(data$contact)

norm.data$y=as.character(norm.data$y) 
norm.data$y[norm.data$y=="no"]= 0
norm.data$y[norm.data$y=="yes"]= 1
str(norm.data$y)


# 3b
# Set the stage for 10 different sets of metrics for odd K's between 3-5.
install.packages("class")
library(class)
set.seed(1)
rep = seq(3,5,2) 
rep.acc = rep
rep.sens = rep
rep.prec = rep

# Create index for 5-fold cross validation
k=5
fold = sample(1:k,nrow(train.x),replace=TRUE)

# Nested for loop
## Outer loop for KNN models with different K
## Inner loop for k-fold cross validation
iter=3 # index for rep iteration
for (K in rep) {
  
  # Space to store metrics from each iteration of k-fold cv
  kfold.acc = 1:k
  kfold.sens = 1:k
  kfold.prec = 1:k
  
  for (i in 1:k) {
    
    # data for test and training sets
    train.kfold = train.x[fold!=i,]
    
    # class labels for test and training sets

    train.cl.actual = train.cl[fold!=i]
    
    # make predictions on class labels for test set
    pred.class = knn(train.kfold,k=K)
    
    # evaluation metrics: accuracy, sensitivity, and precision (for "yes")
    c.matrix = table(test.cl.actual,pred.class)
    acc = mean(train.cl.actual,pred.class)
    sens.yes = c.matrix[1]/(c.matrix[1]+c.matrix[3])
    prec.yes = c.matrix[1]/(c.matrix[1]+c.matrix[2])
    
    # store results for each k-fold iteration
    kfold.acc[i] = acc
    kfold.sens[i] = sens.yes
    kfold.prec[i] = prec.yes
  }
  
  # store average k-fold performance for each KNN model
  rep.acc[iter] = mean(kfold.acc)
  rep.sens[iter] = mean(kfold.sens)
  rep.prec[iter] = mean(kfold.prec)
  iter=iter+1
}

# plot the results for each KNN model.
par(mfrow=c(1,3))
metric = as.data.frame(cbind(rep.acc,rep.sens,rep.prec))
color = c("blue","red","gold")
title = c("Accuracy","Sensitivity","Precision")

for (p in 1:3) {
  plot(metric[,p],type="b",col=color[p],pch=20,
       ylab="",xlab="K",main=title[p],xaxt="n")
  axis(1,at=1:2,labels=rep,las=2)
}

results = as.data.frame(cbind(rep,rep.acc,rep.sens,rep.prec))
names(results) = c("K","accuracy","sensitivity","precision")
results
pred.class = knn(train.kfold,train.cl,k=5)
c.matrix = table(test$default,pred.class)
c.matrix
acc = mean(pred.class==test$default)
sens.yes = c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec.yes = c.matrix[1]/(c.matrix[1]+c.matrix[2])
as.data.frame(cbind(acc,sens.yes,prec.yes))

# 3c
rep = seq(3,5,2) 
rep.acc = rep
rep.sens = rep
rep.prec = rep

# Create index for 5-fold cross validation
k=5
fold = sample(1:k,nrow(train.x),replace=TRUE)

# Nested for loop
## Outer loop for KNN models with different K
## Inner loop for k-fold cross validation
iter=3 # index for rep iteration
for (K in rep) {
  
  # Space to store metrics from each iteration of k-fold cv
  kfold.acc = 1:k
  kfold.sens = 1:k
  kfold.prec = 1:k
  
  for (i in 1:k) {
    
    # data for test and training sets
    test.kfold = train.x[fold==i,]
    
    
    # class labels for test and training sets
    test.cl.actual = train.cl[fold==i]
    
    
    # make predictions on class labels for test set
    pred.class = knn(test.kfold,k=K)
    
    # evaluation metrics: accuracy, sensitivity, and precision (for "yes")
    c.matrix = table(test.cl.actual,pred.class)
    acc = mean(pred.class==test.cl.actual)
    sens.yes = c.matrix[1]/(c.matrix[1]+c.matrix[3])
    prec.yes = c.matrix[1]/(c.matrix[1]+c.matrix[2])
    
    # store results for each k-fold iteration
    kfold.acc[i] = acc
    kfold.sens[i] = sens.yes
    kfold.prec[i] = prec.yes
  }
  
  # store average k-fold performance for each KNN model
  rep.acc[iter] = mean(kfold.acc)
  rep.sens[iter] = mean(kfold.sens)
  rep.prec[iter] = mean(kfold.prec)
  iter=iter+1
}

# plot the results for each KNN model.
par(mfrow=c(1,3))
metric = as.data.frame(cbind(rep.acc,rep.sens,rep.prec))
color = c("blue","red","gold")
title = c("Accuracy","Sensitivity","Precision")

for (p in 1:3) {
  plot(metric[,p],type="b",col=color[p],pch=20,
       ylab="",xlab="K",main=title[p],xaxt="n")
  axis(1,at=1:2,labels=rep,las=2)
}
