> Steps for Random Forest
---
* Pick at random K data points from the Training Set

* Build the Decision Tree associated to these K data points

* Choose the number Ntree of trees you want to build and repeat STEPS 1 & 2

* From new data point, make each one of your Ntree predict the category to which the data points belongs, and assign the new data point
to the category that wins the majority vote
