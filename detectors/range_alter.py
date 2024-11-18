from detectors.utils import Utils
import re
import string
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import chisquare

class RangeDetector:
    def __init__(self, data, coverage_rate=0.89, multiplier=3):
        self.multiplier = multiplier
        self.coverage_rate = coverage_rate
        self.data = [item for item in data if item != '']
        self.train = self.data[:int(0.2*len(self.data))]
        self.test = self.data[int(0.2*len(self.data)):]
        # Get the constraints
        self.categorical_range = self.detect_categorical_range()
        type_, wrangled = Utils.column_type_constraint(self.data)
        if type_ == 'Numerical':
            type_, wrangled = Utils.column_type_constraint(wrangled, True)
            min_range, max_range = self.detect_numerical_range(wrangled, )
            self.numerical_range = [min_range, max_range]
        else:
            self.numerical_range = None
    
        
    # Create histograms and detect categories
    def detect_categorical_range(self):
        # Get, count all the values in the list
        token_info = {}
        for token in self.train:
            if token not in token_info.keys():
                token_info[token] = 1
            else:
                token_info[token] += 1
        
        # Minimize entropy ratio
        sorted_tokens = dict(sorted(token_info.items(), key=lambda item: item[1]))
        # At least half of the tokens
        categories, frequencies = list(sorted_tokens.keys()), list(sorted_tokens.values())
        cut = -1
        max_ratio = 1
        # One or more than one
        if frequencies[-1]/sum(frequencies) > self.coverage_rate:
            return [categories[-1]]
        
        
        for i,f in enumerate(frequencies):
            if i != len(frequencies)-1:
                if frequencies[i+1]/frequencies[i] > max_ratio:
                    cut = i+1
                    max_ratio = frequencies[i+1]/frequencies[i]
        if cut == -1:
            return None
        
        if len(categories)-cut > 16: 
            accepted_categories = categories[-16:]
            accepted_frequencies = frequencies[-16:]
        else: 
            accepted_categories = categories[cut:]
            accepted_frequencies = frequencies[cut:]
        
        test_frequencies = np.zeros(len(accepted_frequencies))
        for token in self.test:
            if token not in accepted_categories:
                pass
            else:
                ind = accepted_categories.index(token)
                test_frequencies[ind] += 1
        # Calculate the chi-square
        statistic, p_value = chisquare(test_frequencies/sum(test_frequencies), np.array(accepted_frequencies)/sum(accepted_frequencies))
        
        # Construct the distribution using the test data
        if p_value <= 0.05: return None
        else: return accepted_categories
    
    def detect_numerical_range(self, numerical_data):
        try:
            data = np.array(numerical_data, dtype=float)
        except ValueError:
            pass  # Handle non-convertible values gracefully
        
        # Calculate mean and standard deviation
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.median(deviations)
        threshold = self.multiplier * mad
        min_range = median - threshold
        max_range = median + threshold
        # Calculate numerical range
        return min_range, max_range