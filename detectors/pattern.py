from .utils import Utils
import re
import string
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

class PatternDetector:
    def __init__(self, data, coverage_rate=0.95):
        self.coverage_rate = coverage_rate
        self.data = pd.Series([str(i) for i in data])

        # Create type mappings and all the combinations
        self.type_mapping = {d:'[0-9]' for d in set(str(i) for i in range(10))}
        self.type_mapping.update({l:'[A-Z]' for l in set(string.ascii_uppercase)})
        self.type_mapping.update({l:'[a-z]' for l in set(string.ascii_lowercase)})

        # Find all the special characters
        special_chars_series = self.data.apply(lambda x: self.extract_special_chars(str(x)))
        self.special_char = list(set().union(*special_chars_series.tolist())) + [' ']
        self.type_mapping.update({s:'.' for s in self.special_char})
        self.data = [item for item in data if item != '']

        # Template info storage
        self.accepted_templates = []
        self.delimiter_bags = []
        self.template_information = {}
        self.pattern_constraints = []
        self.detect_templates()
        self.create_pattern()


    # Filter all the special characters
    def extract_special_chars(self, text):
        return re.findall(r'[^a-zA-Z0-9\s]', text)
    
    
    # Create histograms and detect categories
    def detect_templates(self):
        # Split the templates
        tr_templates = {}
        for item in self.data:
            (template, token_bag, delimiter_bag) = Utils.record_info(self.special_char, str(item))
            # Count the frequency
            if template in tr_templates.keys():
                tr_templates[template]['count'] += 1
            else:
                tr_templates[template] = {'count':1, 'delimiter_bag':delimiter_bag}
        # Sort the templates using length
        sorted_tr_templates = {item[0]: item[1] for item in sorted(tr_templates.items(), key=lambda x: x[1]['count'])}
        # At least half of the tokens
        templates, frequencies = list(sorted_tr_templates.keys()), [v['count'] for v in sorted_tr_templates.values()]
        self.accepted_templates = [templates[0]]
        # Update the delimiter bag
        if len(templates) == 1:
            self.delimiter_bags = [tr_templates[t]['delimiter_bag'] for t in self.accepted_templates]
        
        else:
            self.accepted_templates = templates
            current_freq = [tr_templates[template]['count'] for template in self.accepted_templates]
            # Sort based on the values of list1
            sorted_lists = sorted(zip(current_freq, self.accepted_templates))
            current_freq, self.accepted_templates = (zip(*sorted_lists))
            cut = -1
            max_ratio = 1
            current_freq = [1] + list(current_freq)
            for i,f in enumerate(current_freq):
                if i != len(current_freq)-1:
                    if current_freq[i+1]/current_freq[i] > max_ratio:
                        cut = i
                        max_ratio = current_freq[i+1]/current_freq[i]
            # Update the delimiter bag
            if len(self.accepted_templates) == 1:
                self.delimiter_bags = [tr_templates[t]['delimiter_bag'] for t in self.accepted_templates]     
            elif cut != -1:
                
                if sum(current_freq[cut+1:])/sum(current_freq) > self.coverage_rate:
                    self.accepted_templates = self.accepted_templates[cut:]
                    self.delimiter_bags = [tr_templates[t]['delimiter_bag'] for t in self.accepted_templates]
                else: self.accepted_templates = []
            '''
            elif cut != -1 and sum(current_freq[cut:])/sum(current_freq) > self.coverage_rate:
                self.accepted_templates = self.accepted_templates[cut:]
                self.delimiter_bags = [tr_templates[t]['delimiter_bag'] for t in self.accepted_templates]
            else:
                self.accepted_templates = []
                '''
        
    def match_into_templates(self):
        self.template_information = {self.accepted_templates[i]:{'delimiter_bag':self.delimiter_bags[i], 'records':[[] for j in range(self.accepted_templates[i].count('TOKEN'))]} for i in range(len(self.accepted_templates))}
        sorted_templates = {item[0]: item[1] for item in sorted(self.template_information.items(), key=lambda x: len(x[1]['delimiter_bag']))}
        # From long to short
        sorted_templates_list = list(sorted_templates.keys())[::-1]
        for item in self.data:
            (item_template, token_bag, delimiter_bag) = Utils.record_info(self.special_char, str(item))
            for template in sorted_templates_list:
                # Get the alignment
                if item_template == template:
                    for i, token in enumerate(token_bag):
                        sorted_templates[template]['records'][i].append(token)

    def token_length_constraint_detection(self, split_token_list):
        # Convert to strings to find the length
        token_list = [str(token) for token in split_token_list]
        # Store the information of the token
        lengths = {}
        for token in token_list:
            # Collect the length and the chars
            if len(token) not in lengths.keys():
                lengths[len(token)] = 1
            else:
                lengths[len(token)] += 1
        # Sort the length, return the max
        sorted_length = dict(sorted(lengths.items(), key=lambda item: item[1], reverse=True))
        frequencies = list(sorted_length.values())
        
        if frequencies[0]/sum(frequencies) >= self.coverage_rate:
            return list(sorted_length.keys())[0], True
        else:
            return min(sorted_length.keys()), False
    
    def detect_patterns(self, split_token_list, length_constraint, has_length_constraint):
        # Convert to strings to find the pattern
        token_list = [str(token) for token in split_token_list]
        token_pattern = ''
        # Store the information of the token
        if has_length_constraint: chars = {i:{} for i in range(length_constraint)}
        else: chars = {i:{} for i in range(length_constraint+1)}

        # Constant value(s) detection
        tokens = {}
        for token in token_list:
            if token not in tokens: tokens[token] = 1
            else: tokens[token] += 1
        sorted_tokens = dict(sorted(tokens.items(), key=lambda item: item[1]))
        categories, frequencies = list(sorted_tokens.keys()), list(sorted_tokens.values())
        frequencies = [1] + frequencies
        cut = -1
        max_ratio = 1
        # One or more than one
        if frequencies[-1]/sum(frequencies) > self.coverage_rate:
            return categories[-1]
        for i,f in enumerate(frequencies):
            if i != len(frequencies)-1:
                if frequencies[i+1]/frequencies[i] > max_ratio:
                    cut = i
                    max_ratio = frequencies[i+1]/frequencies[i]
        if cut != -1: 
            if len(categories[cut:]) > 16: 
                cut = -16
            if sum(frequencies[cut+1:])/sum(frequencies) > self.coverage_rate:
                if len(frequencies[cut+1:]) == 1: return categories[cut:]
                else: 
                    return '(%s)'%('|'.join(categories[cut:]))
        '''
        # If we have only one entrance
        if len(tokens.keys()) == 1: return token_list[0]
        kmeans = KMeans(n_clusters=2, n_init='auto')
        token_frequencies = [[value] for value in tokens.values()] + [[1], [sum(tokens.values())]]
        kmeans.fit(token_frequencies)
        labels = kmeans.labels_
        # Collect the frequency
        high_frequency = [token_frequencies[i][0] for i in range(len(token_frequencies)-2) if labels[i] == labels[-1]]
        high_frequency_indices = [i for i in range(len(token_frequencies)-2) if labels[i] == labels[-1]]
        if sum(high_frequency)/sum(tokens.values()) > self.coverage_rate:
            if len(high_frequency) == 1: return list(tokens.keys())[high_frequency_indices[0]]
            else: 
                return '(%s)'%('|'.join([k for i, k in enumerate(tokens.keys()) if i in high_frequency_indices]))
        '''
        for token in token_list:
            for i, char in enumerate(token):
                # In range
                if i in chars.keys():
                    if char not in chars[i].keys(): chars[i][char] = 1
                    else: chars[i][char] += 1
                # Out of range when no length constraint
                elif not has_length_constraint: 
                    if char not in chars[length_constraint].keys(): chars[length_constraint][char] = 1
                    else: chars[length_constraint][char] += 1
        last_type, count = None, 0
        for slot, values in chars.items():
            # Detect static chars
            if len(values.keys()) == 1:
                # Dump last?
                if last_type != None:
                    if count == 1: token_pattern += '%s%s'%(last_type, list(values.keys())[0])
                    elif count > 1: token_pattern += '%s{%d}%s'%(last_type, count, list(values.keys())[0])
                else: token_pattern += '%s'%(list(values.keys())[0])
                last_type, count = None, 0
                # Is this out of minimum range? 
                if not has_length_constraint and slot == length_constraint:
                    token_pattern += '*'
                    return token_pattern
                continue
            else: 
                if max(values.values()) != 1: 
                    # Sort the values
                    sorted_chars = dict(sorted(values.items(), key=lambda item: item[1]))
                    if max(values.values())/sum(values.values()) > self.coverage_rate:
                        token_pattern += '%s'%list(sorted_chars.keys())[-1]
                        last_type, count = None, 0
                        if not has_length_constraint and slot == length_constraint:
                            token_pattern += '*'
                        continue
                    
                # Static type detection
                # Map values to types
                type_maps = {'.':0}
                for key in chars[slot].keys():
                    # Special symbols
                    if key not in self.type_mapping.keys():
                        type_maps['.'] += chars[slot][key]
                    elif self.type_mapping[key] not in type_maps.keys():
                        type_maps[self.type_mapping[key]] = chars[slot][key]
                    else:
                        type_maps[self.type_mapping[key]] += chars[slot][key]
                # Get the types
                sorted_type_maps = {item[0]: item[1] for item in sorted(type_maps.items(), key=lambda x: x[1], reverse=True)}
                types, matches  = [], 0
                for k, v in sorted_type_maps.items():
                    types.append(k)
                    matches += v
                    if matches/sum(sorted_type_maps.values()) >= self.coverage_rate:
                        break
                # Construct the current type
                if len(types) == 1:
                    current_type = types[0]
                elif len(types) > 1 and '.' not in types:
                    current_type = '['
                    for item in types:
                        if item == '[0-9]':
                            current_type += '0-9'
                        elif item == '[a-z]':
                            current_type += 'a-z'
                        elif item == '[A-Z]':
                            current_type += 'A-Z'
                    current_type += ']'
                else:
                    current_type = '.'
                # Check whether current type equals to the last type
                if current_type == last_type:
                    if has_length_constraint or slot < length_constraint:
                        count += 1
                    else:
                        # Append the last and return
                        if count == 1: token_pattern += '%s+'%(last_type)
                        else: token_pattern += '%s{%d,}'%(last_type, count)
                        return token_pattern
                else:
                    if last_type != None:
                        if count == 1: token_pattern += '%s'%(last_type)
                        elif count > 1: token_pattern += '%s{%d}'%(last_type, count)
                    if not has_length_constraint and slot == length_constraint:
                        # Append current and return
                        token_pattern += '%s*'%(current_type)
                        return token_pattern
                    last_type = current_type
                    count = 1

        # Dump the last?
        if last_type != None:
            if count == 1: token_pattern += '%s'%(last_type)
            elif count > 1: token_pattern += '%s{%d}'%(last_type, count)
        return token_pattern
    
    def create_pattern(self):
        self.detect_templates()
        self.match_into_templates()
        if self.accepted_templates != None:
            for template, values in self.template_information.items():
                composed_template = template
                # Parse the token list
                for i in range(len(values['records'])):
                    # Check type and filter (int, str)
                    type_count = [0, 0]
                    # Check the types
                    numerical_column = [] 
                    string_column = []
                    for record in values['records'][i]:
                        try:
                            int(record)
                            numerical_column.append(record) 
                            type_count[0] += 1

                        except ValueError:
                            string_column.append(record)
                            type_count[1] += 1
                    type_count = [type_count[0]/len(values['records'][i]), type_count[1]/len(values['records'][i])]
                    if max(type_count) < self.coverage_rate:
                        wrangled_tokens = values['records'][i]
                    elif type_count[0] > type_count[1]:
                        wrangled_tokens = numerical_column
                    else:
                        wrangled_tokens = string_column
                    # Get and test pattern constraints
                    length_constraint, has_length_constraint = self.token_length_constraint_detection(wrangled_tokens)
                    pattern_constraint = self.detect_patterns(wrangled_tokens, length_constraint, has_length_constraint)
                    # Which one to take?
                    composed_template = re.sub('TOKEN', r'%s'%pattern_constraint, composed_template, 1)
                self.pattern_constraints.append(composed_template)
