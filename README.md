# ML-Projects

My personal machine learning learning repository where I implement projects to understand ML concepts hands-on.

---

## What's This?

This is my collection of ML projects that I've worked on while learning machine learning. Each project helped me understand different concepts, from building a neural network from scratch to working with real-world datasets.

The goal isn't perfection—it's learning by doing. Some projects are cleaner than others, and that's okay. They all represent different stages of my learning journey.

---

## Projects

### Neural Network from Scratch
**📁 `Neural_Network_Scratch/`**

Built a complete neural network using only NumPy to really understand what's happening under the hood. No TensorFlow, no PyTorch—just math and code.

What I learned:
- How backpropagation actually works (not just the theory)
- Why weight initialization matters
- The mechanics behind gradient descent
- How activation functions affect learning

Later turned this into a more complete Java library: [TinyNN-Java](https://github.com/GEMIv1/TinyNN-Java)

---

### Boston House Price Prediction
**📁 `boston-house-pricing-prediction/`**

Classic regression problem. Predicting house prices based on features like crime rate, rooms, and location.

What I learned:
- How to explore and visualize data
- Feature correlation and importance
- Comparing different regression models
- When regularization helps

---

### Fraud Detection
**📁 `fraud-detection/`**

Working with highly imbalanced data where fraudulent transactions are rare. This was challenging because normal accuracy metrics don't work well here.

What I learned:
- How to handle class imbalance (SMOTE, undersampling)
- Why precision and recall matter more than accuracy
- ROC curves and AUC scores
- Dealing with real-world messy data

---

### Titanic Survival Prediction
**📁 `titanic/`**

The classic Kaggle starter competition. Good for practicing data cleaning and feature engineering.

What I learned:
- Handling missing data
- Feature engineering (creating new features from existing ones)
- Encoding categorical variables
- Basic classification techniques

---

### Used Cars Price Prediction
**📁 `used-cars-price-prediction/`**

Predicting used car prices based on brand, year, mileage, and other features.

What I learned:
- Working with categorical data at scale
- Dealing with outliers
- Non-linear relationships in data
- Tree-based models

---

## Tech Stack

- **NumPy** - Array operations and math
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - ML algorithms and tools
- **Jupyter Notebook** - Interactive coding

---

## Setup

If you want to run these notebooks:

```bash
# Clone the repo
git clone https://github.com/GEMIv1/ML-Projects.git
cd ML-Projects

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter scipy

# Start Jupyter
jupyter notebook
```

Then just open any `.ipynb` file in the project folders.

---

## Structure

```
ML-Projects/
├── Neural_Network_Scratch/       # Neural net from scratch
├── boston-house-pricing-prediction/
├── fraud-detection/
├── titanic/
├── used-cars-price-prediction/
└── README.md
```

Each folder contains the notebook(s) and data for that project.

---

## Learning Path

If you're also learning ML and want to follow a similar path:

1. **Start with Titanic** - It's beginner-friendly and teaches the basics
2. **Then Boston Housing** - Good intro to regression
3. **Try Used Cars** - More feature engineering practice
4. **Challenge yourself with Fraud Detection** - Real-world complexity
5. **Finally Neural Network from Scratch** - Deep dive into fundamentals

---

## Notes

- These projects are learning exercises, not production code
- Some notebooks might be messy—that's part of the learning process
- I'm still learning, so if you see something that could be improved, feel free to open an issue or PR
- The code evolves as I learn better practices

---

## Related Work

After building the neural network from scratch in Python, I reimplemented it in Java as a proper library: [TinyNN-Java](https://github.com/GEMIv1/TinyNN-Java)

---

**If you're also learning ML and find these helpful, feel free to star the repo or reach out!**

[@GEMIv1](https://github.com/GEMIv1)
