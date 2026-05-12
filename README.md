# 🌳 High-Income Potential Prediction with Decision Trees & Random Forests

A machine learning project using the 2024 CPS ASEC dataset to predict whether an individual earns more than **$50K annually**, with a focus on tree-based models, nonlinear pattern detection, pruning strategies, and model interpretability.

This project was developed as part of a Machine Learning course team project. My primary contribution focused on the **Decision Tree and Random Forest modeling workflow**, including baseline modeling, preprocessing experiments, pruning, hyperparameter tuning, result comparison, and feature importance interpretation.

---

## 🚀 Project Overview

Income potential is shaped by multiple interacting factors, including education, age, working hours, industry, marital status, nativity, race, and sex. Because these relationships are not always linear, this project explores how tree-based machine learning models can capture nonlinear patterns and feature interactions in socioeconomic data.

The goal was to answer two questions:

1. Can we predict whether a person earns more than $50K per year?
2. Which factors are most important in driving high-income potential?

---

## 🧠 My Role

I focused on the tree-based modeling section of the project:

- Built baseline **Decision Tree** models
- Tested preprocessing strategies for categorical variables
- Compared raw, grouped, and dummy-encoded feature versions
- Applied **pre-pruning** through `max_depth` tuning
- Applied **post-pruning** using cost-complexity pruning
- Built and tuned **Random Forest** models
- Compared model performance using accuracy, precision, recall, and F1-score
- Interpreted feature importance to explain key income drivers

This part of the project helped me strengthen my understanding of how Decision Trees and Random Forests handle nonlinear classification problems, especially when working with mixed numerical and categorical features.

---

## 📊 Dataset

**Source:** 2024 CPS ASEC  
**Task:** Binary classification  
**Target:**  
- `1` = annual income > $50K  
- `0` = annual income ≤ $50K  

The dataset was filtered to focus on active working adults:

- Age > 16
- Hours worked per week > 0
- Total annual income > $100

After filtering, the modeling dataset included approximately **64K working adults**.

---

## 🛠️ Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

---

## 🔍 Modeling Workflow

### 1. Baseline Decision Tree

I first trained a raw Decision Tree model without extensive preprocessing. Although the raw model produced relatively high accuracy, it relied heavily on a single dominant feature: `hours_per_week`.

This revealed an important modeling issue:

> High accuracy does not always mean strong generalization.

The model appeared to overfit noise-driven patterns and showed weaker recall for identifying high-income individuals.

---

### 2. Preprocessing Experiments

To improve model stability and interpretability, I tested several preprocessing strategies:

- Grouped marital status into broader categories
- Reduced high-cardinality nativity values into top countries plus “Other”
- Grouped race categories
- Converted detailed education codes into meaningful education tiers
- Compared grouped categorical features against dummy encoding

The goal was not simply to increase accuracy, but to build a model that produced more balanced, interpretable, and generalizable results.

---

### 3. Pre-Pruning

I tested multiple Decision Tree depths:

- `max_depth = 3`
- `max_depth = 5`
- `max_depth = 7`
- `max_depth = None`

The results showed a clear bias-variance trade-off:

- Very shallow trees underfit the data
- Fully grown trees overfit the training data
- A controlled tree depth produced the best balance between interpretability and performance

The best-performing Decision Tree used:

```python
max_depth = 7
````

---

### 4. Post-Pruning

After selecting the best pre-pruned tree, I applied cost-complexity pruning using `ccp_alpha`.

Post-pruning helped simplify the tree by removing weak or redundant branches. The pruned tree became easier to interpret while maintaining, and slightly improving, test performance.

This step demonstrated how pruning can reduce overfitting without sacrificing model usefulness.

---

### 5. Random Forest Modeling

I then extended the single-tree approach into a Random Forest model.

Random Forest was useful because it reduces the high variance of individual Decision Trees by averaging predictions across many trees.

I tuned key hyperparameters:

* `max_depth`
* `n_estimators`

The best Random Forest model used:

```python
max_depth = 9
n_estimators = 300
```

Random Forest achieved the strongest overall performance, with an F1-score of approximately **0.775**.

---

## 📈 Model Comparison

| Model                | Key Strength                                              | Key Limitation                                    |
| -------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| Logistic Regression  | Simple and interpretable linear baseline                  | Limited ability to capture nonlinear interactions |
| Decision Tree        | Interpretable nonlinear rules                             | Sensitive to overfitting                          |
| Pruned Decision Tree | Better generalization and cleaner rules                   | Still less stable than ensemble methods           |
| Random Forest        | Best predictive performance and stable feature importance | Less directly interpretable than a single tree    |

Random Forest performed best because it captured nonlinear relationships while reducing the instability of a single Decision Tree.

---

## 🔑 Key Findings

Across Decision Tree and Random Forest models, the most important predictors were:

1. **Education level**
2. **Hours worked per week**
3. **Age**
4. **Sex**
5. **Nativity and marital status**
6. **Industry**

The results suggest that high-income potential is strongly associated with human capital, labor intensity, and career-stage factors.

---

## 💡 What I Learned

This project helped me understand that tree-based modeling is not just about fitting a model and reading accuracy.

The most important lessons were:

* Decision Trees are powerful for nonlinear classification, but easy to overfit
* Preprocessing changes how trees split and interpret features
* Accuracy can be misleading when recall and F1-score tell a different story
* Pruning is essential for balancing interpretability and generalization
* Random Forest improves stability by reducing variance across individual trees
* Feature importance can connect technical model outputs back to business interpretation

---

## ⚠️ Limitations

This project has several limitations:

* The model uses one year of CPS ASEC data
* Important income-related features such as occupation, job tenure, region, employer quality, and skills were not included
* The target is simplified into a binary threshold of $50K
* Demographic variables such as sex and race require careful ethical interpretation
* Random Forest improves performance but is less transparent than a single Decision Tree

---

## 📁 Repository Structure

```text
.
├── notebooks/
│   ├── 01_project_overview.ipynb
│   └── 02_dt_rf_code_showcase.ipynb
├── data/
│   └── README.md
├── outputs/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── tree_visualization.png
├── README.md
└── requirements.txt
```

---

## 📌 Portfolio Highlights

This project demonstrates my ability to:

* Build machine learning models for real-world classification problems
* Use Decision Trees and Random Forests to capture nonlinear relationships
* Diagnose overfitting through performance comparison
* Apply pruning and hyperparameter tuning
* Translate model results into business and socioeconomic insights
* Communicate technical findings clearly for a non-technical audience

---

## 🔗 Links

* Project Notebook: [Link]
* Code Showcase Notebook: [Link]
* Presentation / Report: [Link]
* Portfolio Page: [Link]

---

## ✨ Final Takeaway

The strongest model was Random Forest, but the most valuable part of the project was the modeling process itself: starting from a raw Decision Tree, identifying overfitting, improving preprocessing, controlling tree complexity, and finally using ensemble learning to build a more stable and generalizable classifier.

```
```
