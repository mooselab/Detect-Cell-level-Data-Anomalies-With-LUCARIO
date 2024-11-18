from collections import Counter
import numpy as np
import re

class Utils:
    # Entropy calculation
    def entropy(probabilities):
        entropy_value = 0
        for prob in probabilities:
            if prob != 0:  # To avoid log(0)
                entropy_value -= prob * np.log2(prob)
        return entropy_value
    
    def list_contains_in_order(a, b, return_alignment=False):
        alignment = []
        # Initialize indices for list a and list b
        i, j = 0, 0
        
        # Iterate through list a and list b simultaneously
        while i < len(a) and j < len(b):
            # If the current element in list a matches the current element in list b
            if a[i] == b[j]:
                # Move to the next element in list b
                j += 1
            # Move to the next element in list a
            i += 1
            alignment.append(j)
        
        # If we reached the end of list b, it means all elements of b were found in a in order
        if return_alignment:
            return j == len(b), alignment
        else:
            return j == len(b)

    # Collect the character information of a list of data
    def record_info(special_char, record):
        template = ''
        token = ''
        delimiter = ''
        token_bag, symbol_bag = [], []
        # Go through the record to collect the tokens
        for char in record:
            # Collect the full token
            if char in special_char:
                delimiter += re.escape(char)
                if token != '':
                    token_bag.append(token)
                    token = ''
            else:
                if delimiter != '':
                    symbol_bag.append(delimiter)
                    template += delimiter
                    delimiter = ''
                if token == '':
                    template += 'TOKEN'
                    token += char
                # continue counting the token itself
                else:
                    token += char
                
        # Dump the things that are not dumped
        if token != '':
            token_bag.append(token)
        if delimiter != '':
            symbol_bag.append(delimiter)
            template += delimiter
        return (template, token_bag, symbol_bag)
            
    def column_type_constraint(column, conversion=False):
        # float, integer& string
        # Third type: mixed
        type_count = [0, 0]
        # Check the types
        numerical_column = [] 
        string_column = []
        for record in column:
            try:
                float(record)
                # Convert to the new type?
                if conversion:
                    numerical_column.append(float(record)) 
                else:
                    numerical_column.append(record) 
                type_count[0] += 1
                
            except ValueError:
                string_column.append(record)
                type_count[1] += 1
        e = Utils.entropy([type_count[0]/len(column), type_count[1]/len(column)])
        # The highest entropy can be reached by the types
        # Return types and the corrected column
        if e >= 0.5:
            return 'Mixed', column
        elif type_count[0] > type_count[1]:
            return 'Numerical', numerical_column
        else:
            return 'String', string_column
        
    def type_anomaly_detector(record, column_type):
        # Check types
        if column_type == 'Numerical':
            try:
                float(record)
                return False
            except ValueError: return True
        # Check types
        elif column_type == 'String':
            try:
                float(record)
                return True
            except ValueError: return False
        else:
            return False