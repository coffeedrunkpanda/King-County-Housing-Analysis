import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(X):
    # 1. Calculate correlation
    corr = np.round(np.abs(X.corr()), 2)
    
    # 2. Create the mask (True for upper triangle and diagonal)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # 3. Slice the matrix and mask to remove the completely empty row/column
    #    Row 0 is fully masked (hidden), Column -1 is fully masked (hidden)
    corr_sliced = corr.iloc[1:, :-1]
    mask_sliced = mask[1:, :-1]

    f, ax = plt.subplots(figsize=(16, 16))

    # 4. Plot using the sliced data
    #    Note: annot=True is safer than annot=corr when slicing
    sns.heatmap(
        corr_sliced, 
        mask=mask_sliced, 
        annot=True,          # Use True to automatically label values
        square=True, 
        linewidths=.5, 
        vmax=1
    )
    plt.show()