import numpy as np
from scipy.stats import mode


from scipy.stats import mode


def handle_outliers(data, upper_bound, lower_bound, method, custom_value=None):
    """
    Function to handle outliers in a numpy array.

    Parameters:
    data: numpy array - the input array.
    method: str - the method to handle outliers. Possible values are:
        - 'mean': replace with mean of non-outlier values.
        - 'mode': replace with mode of non-outlier values.
        - 'median': replace with median of non-outlier values.
        - 'windsorize': replace with the closest non-outlier value within bounds.
        - 'custom': replace with user-defined value.
    custom_value: float - the value to use when method='custom'.

    Returns:
    numpy array - the modified array with outliers handled.
    """
    # Calculate the lower and upper bounds for outliers
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lb = q1 - (lower_bound * iqr)
    ub = q3 + (upper_bound * iqr)

    # Determine the non-outlier values
    non_outlier_mask = (data > lb) & (data < ub)
    outliers_mask = (data < lb) | (data > ub)
    outliers = data[outliers_mask]
    non_outlier_values = data[non_outlier_mask]
    print(data[non_outlier_mask])
    outlier_replacements = np.empty(data.shape)

    # Replace outliers using the selected method
    if method == 'mean':
        mean = outliers.mean()
        print('mean', mean)
        outlier_replacements.fill(mean)
    elif method == 'mode':
        outlier_replacements.fill(mode(non_outlier_values, keepdims=True)[0][0])
    elif method == 'median':
        outlier_replacements.fill(outliers.median())
    elif method == 'windsorize':
        for i, val in enumerate(data):
            if val < lb:
                if len(non_outlier_values) > 0:
                    outlier_replacements[i] = non_outlier_values[np.argmin(np.abs(outliers - lb))]
                else:
                    outlier_replacements[i] = val
            elif val > ub:
                if len(non_outlier_values) > 0:
                    outlier_replacements[i] = non_outlier_values[np.argmin(np.abs(outliers - ub))]
                else:
                    outlier_replacements[i] = val
            else:
                outlier_replacements[i] = val
    elif method == 'custom':
        outlier_replacements.fill(custom_value)

    # Replace outliers with the selected
    data[outliers_mask] = outlier_replacements[outliers_mask]
    return data

np.random.seed(123)
data = np.random.normal(0, 1, 1000)
data[0] = 10
data[1] = -10
data[2] = 20

# Handle outliers with the 'mean' method
new_data = handle_outliers(data=np.array([-10, -100,1,2,2,3,4,5,18]), method='mean', lower_bound=1.5, upper_bound=1.5)
print(new_data)