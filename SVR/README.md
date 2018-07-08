
# Computer Hardware Performance Prediction

I have used a dataset available on UCI repository to predict the relative performance of computer hardware based on some features like
1. Name of the vendor
2. Cach memory
3. Minimum main memory in kilobytes
4. Maximum main memory in kilobytes
5. Minimum channels in units
6. Maximum channels in units
and by using this information, we have to predict the relative performance of the hardware.

I have used SVM Regressor for this. There are number of advantages of this model. You can read more about this on scikit-learn offical website here
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
Some more resources are as follows:
1. https://www.youtube.com/watch?v=dIIIa0mljJo
2. https://www.youtube.com/watch?v=8qsFI22c5Lk
3. https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
4. https://sadanand-singh.github.io/posts/svmpython/

So what are the application of such model. Well, lot of hardware manufacturing companies like Nvidia, intel and IBM can use such models to compare their hardware with other companies to follow the best design process. The data that I have used here is not enough to do that but we can have lot more information from the original creator of the hardware.

