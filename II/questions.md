1. LinearSVC vs SVC:

Is LinearSVC appropriate for this credit default problem?
What predictive power do we lose by using linear decision boundary? We have tried the normal svc, it is extremely slow

2. Cost-Sensitive Learning Strategy:
For linear regression, should I optimize threshold after training (post-hoc)?
Or should I use custom loss during GridSearchCV?

3. Industry best practice for credit risk modeling?
Business Cost Structure:
Is FN:FP = 10:1 a reasonable cost ratio for credit default?
How should this ratio be determined in practice? we have the data, should we calculate the average of based on real default/no default?

4. Model Selection Criteria:
Currently selecting best model by AUC on validation set, with the standard loss function (L2, hinge and logistic)
Should I select by business cost instead? 
How to balance statistical performance vs business objective?

5. Feature Engineering Validation:
Are the 8 engineered features (slopes, ratios, etc.) appropriate? should we select more? residual analysis based on the class?

6. Some functions have the imbalanced function when training the data, should we explicitely deal with the imbalance ourselves (undersampling or oversampling, try different method and compare validation cost?)
  
7. Three-Tier Decision System, so when we have the dicision also have the strategy, how we should evaluate the performance of our decision? model efficiency, and business efficiency would be different. Per zone accuracy? 

As a business enhancement, we extend to a three-zone lending strategy:
High-risk (P ≥ τ_high): Automatic Rejection
Medium-risk (τ_low < P < τ_high): Manual Underwriting Review
Low-risk (P ≤ τ_low): Automatic Approval

8. Some bonus for revisiting Project 1:
Is your chosen portfolio better than LASSOCV or RidgeCV using new data?

Can we just resubmit the first project with update or we have to redo the first project with this requirement in order to get the bonus?
