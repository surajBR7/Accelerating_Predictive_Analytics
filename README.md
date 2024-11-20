# Accelerating Predictive Analytics on Large Datasets through Embarrassingly Parallel Computing

This project demonstrates how predictive analytics on large datasets can be accelerated using embarrassingly parallel computing techniques. The study uses a **GradientBoostingClassifier** to model data and applies parallel processing using Python's `multiprocessing` library to speed up computations.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Parallel Processing](#parallel-processing)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Overview
In this project, we investigate the tradeoff between data size and processing time. Predictive models often face challenges when the dataset size increases, leading to longer processing times. By using parallel computing, we can speed up the process of building and evaluating machine learning models.

The study focuses on balancing time and data size to achieve optimal performance for predictive analytics. Using the Python programming language and the `multiprocessing` library, this project compares serial and parallel execution for processing large datasets.

### **Key Features**
1. **Data Import and Preprocessing:** 
   - Large datasets are preprocessed for efficient analytics.
2. **Model Implementation:**
   - The models are optimized to take advantage of multi-core systems.
3. **Use of Parallel Processing:**
   - Parallelization is implemented using Python’s `multiprocessing` library and the `Pool()` module.

## Project Structure
```plaintext
.
├── main.py              # Contains the main implementation of data preprocessing and model training
├── model.py             # Defines the GradientBoostingClassifier and parallel processing functions
├── Presentation.pptx    # Project presentation explaining methods and results
├── data                 # Folder for input data
├── README.md            # This file
```
## Data 

The dataset used contains 100,000 rows and 10 columns. After preprocessing, we selected four features: num_pages, book_rating, book_price, and text_lang. The target variable is book_genre.

To ensure that no single feature dominates the model, we used a standard scaler to normalize the data.

## Model
We utilized the GradientBoostingClassifier for classification, which builds decision trees sequentially. The settings used for training include:
  - n_estimators = 100
  - learning_rate = 0.1
    
The classifier was applied to the dataset and predictions were made based on the highest predicted probabilities.

## parallel Processing

### Serial Processing

In serial processing, tasks are executed one at a time. This leads to high workload for the processor and longer execution times.

### Parallel Processing

By using Python's multiprocessing library, we implemented parallel processing where multiple tasks are completed simultaneously across different processor cores. This approach significantly reduces the time required to process large datasets.

### Steps for Parallelization:
  - Import and preprocess the data.
  - Train the GradientBoostingClassifier model.
  - Apply multiprocessing with different configurations of cores and data sizes.
  - Measure time taken, speedup, and efficiency for each configuration.

## Result
Parallel processing demonstrated a noticeable speedup, especially for larger datasets. By distributing tasks across multiple cores, we reduced the time required to make predictions while maintaining model accuracy.
#### **12 Cores**
- **Data size:** 1,000  
  - **Time (Serial):** 0.5026s  
  - **Time (Parallel):** 0.2988s  
  - **Speedup:** 1.68  
  - **Efficiency:** 7.01%  
- **Data size:** 50,000  
  - **Time (Serial):** 24.9746s  
  - **Time (Parallel):** 2.0411s  
  - **Speedup:** 12.24  
  - **Efficiency:** 50.98%

#### **6 Cores**
- **Data size:** 1,000  
  - **Time (Serial):** 0.5003s  
  - **Time (Parallel):** 0.3551s  
  - **Speedup:** 1.41  
  - **Efficiency:** 5.87%  
- **Data size:** 50,000  
  - **Time (Serial):** 24.1345s  
  - **Time (Parallel):** 1.9939s  
  - **Speedup:** 12.10  
  - **Efficiency:** 50.43%

#### **24 Cores**
- **Data size:** 1,000  
  - **Time (Serial):** 0.7596s  
  - **Time (Parallel):** 0.1892s  
  - **Speedup:** 4.02  
  - **Efficiency:** 16.73%  
- **Data size:** 50,000  
  - **Time (Serial):** 38.0530s  
  - **Time (Parallel):** 2.5239s  
  - **Speedup:** 15.08  
  - **Efficiency:** 62.82%

---

## **Technologies Used**
- **Python Libraries:** 
  - `multiprocessing`
  - `pandas`
  - `matplotlib`
  - `seaborn`
- **Supercomputing Resources:** Multi-core processors (6, 12, and 24 cores).
  
---
## How to run this:
- Clone the repository:
  ```bash
  git clone https://github.com/yourusername/your-repo-name.git
  cd your-repo-name
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
   ```
- Run the program:
  ```bash
  python3 main.py
  ```

## References
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Towards Data Science: Parallelization in Python](https://towardsdatascience.com/parallelization-w-multiprocessing-in-python-bd2fc234f516)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## Author
- Suraj Basavaraj Rajolad



  
