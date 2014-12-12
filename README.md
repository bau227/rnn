CS 221 Project: Recurrent Neural Networks for Stock Price Prediction.

December 2014

Stanford -- Ben and Anwar

Most of the code is experimental and a lot of it is interactive. This is by no means a single application but several testing functions that are undocumented and do not include any error handling. We are trying to test different ML algorithms to see which one works better in predicting financial time-series data (stock quotes). The algorithms are: Logistic Regression (LR); Support Vector Machines (SVM) with Linear, rbf, and polynomial kernels; Feed-Forward Neural Network (FF-NN), and Recurrent Neural Network (RNN). Most of the output is then transferred to Excel for analysis.

Having said that, we have refactored a good portion of the code. You can find it in rnn_demo, rnn_features, rnn_baseline. These files are documented to some degree. In you type python rnn_demo.py the program will create a dataset for all the quotes in the quotes directory, and will run LR tests, SVM linear kernel, SVM rbf kernel, on these quotes for 1 day, 3 days, 7 days, 11 days (mid-month), and 22 days (month). Before finishing, it will also load a sample dataset and save a .train version and a .test version. You don't need any external lib to run rnn_demo (numpy is required), and make sure that svm_lib.py is in the same directory.

3 directories are required for demo: ./quotes, ./data, ./train

Make sure that they are in the same directory as rnn_xxx. The directory quotes contains the raw data (csv), while ./data and ./train may contain some previous data but you can delete them. Demo overwrites the files in these folders. You can modify rnn_demo.py and change the path and names of these directories.

That's it. Good luck. Thank you...

