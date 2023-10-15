<ins>**About the Dataset**</ins>

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. This dataset is commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.

<br/>
<br/>

<ins>**Purpose**</ins>

The main purpose of this project was to use data science to make intelligent business decisions based on a Kaggle data set. Data mining models were created to predict if a loan applicant should be approved or rejected based on theirnumber of dependents, education level, emplyment status, annual income, requested loan amount, requested loan term, cibil score (FICO score), and monetary assets. The following models were created during this project:

- Logistic Regression
- K-Nearest Neighbor
- Gradient Boosting
- Logistic Regression
- Decision Tree
- Random Forest
- Stachastic Gradient Descent
- Neural Network
- Hyperparameter Tuning with GridSearchCV
<br/>

<ins>**Results**</ins>

To evaluate the results for this project it is necessary to determine a baseline model. A cumulative gains chart was used to plot the accuracy of each model compared to the baseline model. Here it was easy to visualize how different models compared to the baseline model.
<br/>
<br/>

<img width="514" alt="image" src="https://github.com/dsklnr/Loan_Prediction/assets/101298501/7d9ea37d-fdf0-4e77-8f1f-c143e5d4162e">
<br/>
<br/>

The model with the highest accuracy was the decision tree, with an accuracy of 97.44%, precision of 97.67%, recall of 98.18% and an F1-score of 97.92%. This implies that tree-based models, such as the Random Forest and the Decision Tree, were the best models, in terms of accuracy for predicting loan approvals based on the given features. The Random Forest's results show how powerful ensemble models can be in finding patterns in datasets with many different features and relationships such as ours. The logistic regression models, KNN model and gradient boosting models also had impressive accuracy and could be preferred models due to their simplicity and ease of explainability in results. On the other hand, while the Neural Network's high accuracy also displays the flexibility and potential of deep learning models, results and predictions can be difficult to explain and interpret, which could be very important whether in explaining results to customers or management . It should also be noted that, at an accuracy that's not too far off, the SGD model could also be a preferred choice when the data is extremely large or if there are limited computing resources.

Machine learning can create new innovations and transformations for financial institutions, especially banks, to update their traditional loan approval processes. Traditional methods are often time-consuming and can be prone to human biases and overlook data patterns that algorithms can accurately find (Roth, 2023). Automating the loan approval process through exploratory data analysis and predictive models, such as the ones created in this project, not only makes this decision-making process faster, leading to cost savings and higher customer satisfaction, but can also lead to lower risk of loans and higher likelihood of returns from the borrowers.

The high accuracy of these models can ensure a very detailed risk assessment of every loan. By finding these hidden patterns in applicant data, banks can make more informed lending decisions and can discern between higher and lower risk individuals. This accuracy can reduce the likelihood of defaults and can create a higher return on investment. In summary, implementing machine learning into the traditional loan approval process can lead to faster decision making, higher efficiency, lower risk, and increased profitability for institutions such as banks.


<br/><br/>
<ins>**Link to Kaggle**</ins>

https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
