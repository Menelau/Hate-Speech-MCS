# Selecting and combining complementary feature representations and classifiers for hate speech detection

Official implementation of the paper "Selecting and combining complementary feature representations and classifiers for hate speech detection" by Rafael M. O. Cruz, Woshington V. de Souza and George D. C. Cavalcanti.

Code is organized as follows:

- The processing.py file content the functions responsible for pre-processing of the tweets.
- The function.py file was created to centralize functions such as calculating the dissimilarity matrix and reducing dimensionality.
- The Monolithic.py contains the code to train and optimize hyperparameters (grid search) for the 8 classification algorithms considered in this work.
- The jupyter notebooks that start with "Experiment" are the files of the experiments for each dataset used to build the paper.
- The folder "Save Predict and Proba" have the preprocessed predictions and probabilities obtained by each feature extraction method and classification algorithm trained for each dataset.
- Diversity analysis.ipynb contains the code to compute the diversity between the models and generate the Classifier Projection Space (CPS) to analyze the relationship between the classifiers.
- Stacking.ipynb contains the code to apply the Stacked Generalization method to combine the trained models.

Requirements and installation:
------------------------------
This code requires Python >= 3.6.5, Zeugma, Scikit-learn, DESlib, Pandas, NLTK and Matplotlib. Environment can be installed using the following command:

pip install requirements.txt -r

References:
-----------
[1] : Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between combination methods and measures of diversity in combining classifiers." Information fusion 3.2 (2002): 135-148.

[2] : L. I. Kuncheva, Combining Pattern Classifiers: Methods and Algorithms, Wiley-Interscience, 2004.

[3] Pękalska, Elżbieta, Robert PW Duin, and Marina Skurichina. "A discussion on the classifier projection space for classifier combining." In International workshop on multiple classifier systems, pp. 137-148. Springer, Berlin, Heidelberg, 2002.

[4] : Wolpert, David H. "Stacked generalization." Neural networks 5, no. 2 (1992): 241-259.

[5] Cruz, Rafael MO, George DC Cavalcanti, Ren Tsang, and Robert Sabourin. "Feature representation selection based on classifier projection space and oracle analysis." Expert Systems with Applications 40, no. 9 (2013): 3813-3827.

[6] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

[7] : Cruz, Rafael MO, Luiz G. Hafemann, Robert Sabourin, and George DC Cavalcanti. "DESlib: A Dynamic ensemble selection library in Python." Journal of Machine Learning Research 21, no. 8 (2020): 1-5.
