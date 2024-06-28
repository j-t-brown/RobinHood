"""
Code to run OLA global k-anonymity generalization.
Precursor for Robin Hood masking.
"""

## Libraries
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from functools import partial
from multiprocessing import Pool


## Helper functions

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 1e-10
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def process_group_object(data, attributes, generalization_hierarchies, k, suppression_prop, name_hier,
 group_var, divergence_prior, group_value):
    
    # create standard OLA object on subset of the data
    obj = OLA(data = data[data[group_var] == group_value],
              attributes = attributes,
              generalization_hierarchies = generalization_hierarchies,
              k = k,
              suppression_prop = suppression_prop,
              name_hier = name_hier,
              group_var = group_var,
              divergence_prior = divergence_prior)

    # find k minimal generalizations
    obj.find_kmin()

    # find optimal k minimal generalizations for each utility measure
    obj.process_kmin()


    # return completed object and group value identifier
    return [group_value, obj]


## OLA classes

class OLA_histogram:
    
    
    def __init__(self, data, attributes, generalization_hierarchies, generalization_types, k, suppression_prop, name_hier,
        group_var, util_functions = ['nonu_ent'], ignore_cols = [], divergence_prior = 1e-10, adjust_prec_dict=False):
        
        # store inputs
        self.data = data.copy()
        self.attributes = attributes.copy()
        self.gen_hiers = generalization_hierarchies.copy()
        self.gen_type = generalization_types.copy()
        self.k = k
        self.sup_prop = suppression_prop
        self.tot_pop = data.counts.sum()
        self.name_hier = name_hier.copy()
        self.util_functions = util_functions
        self.prior = divergence_prior
        self.group_var = group_var
        self.done = False
        self.ignore_cols = ignore_cols
        self.adjust_prec_dict = adjust_prec_dict

        # check for missing attributes
        for attr in self.attributes:
            if attr not in self.data.columns:
                raise ValueError('{} not in provided dataset'.format(attr))

        for attr in self.attributes:
            if attr not in list(self.gen_hiers.keys()):
                raise ValueError('{} not in generalization hierarchies'.format(attr))

        for attr in self.attributes:
            if attr not in list(self.name_hier.keys()):
                raise ValueError('{} not in name hierarchy'.format(attr))

        # check for too many attributes
        for col in self.data:
            if (col not in self.attributes) & (col not in ['counts']) & (col not in ignore_cols):
                raise ValueError('{} not in specified attributes'.format(col))
        
        # initialize lattice
        self.top_node = tuple([max(self.gen_hiers[hier].keys()) for hier in self.gen_hiers.keys()])
        self.bottom_node = tuple([0 for _ in self.attributes])
        self.initialize_lattices()
        
        # probabilities for divergence calculation
        #self.initialize_div_probs()
        
        
    def initialize_div_probs(self):
        self.data['counts_prior'] = self.data.counts + self.prior
        self.prob_pop = self.data.counts_prior.sum()
        self.P = self.data.counts_prior / self.prob_pop
        self.group_totals = (self.data.groupby(self.group_var, sort=False)['counts_prior']
                             .transform('sum')).values
        self.group_P = np.divide(self.data.counts_prior, self.group_totals)
        
        # group total counts without prior
        self.group_counts = self.data.groupby(self.group_var).agg({'counts':'sum'})
    

    def set_group_var(self, group_var):
        self.group_var = group_var
        self.initialize_div_probs()
        
    
    def initialize_lattices(self):
        
        # full latice
        self.full_lattice = defaultdict(dict)
        possible_values = (range(self.bottom_node[i], self.top_node[i] + 1) for i in range(len(self.top_node)))
        
        for node in itertools.product(*possible_values):
            self.full_lattice[sum(node)][node] = None
            
        # k-minimal results
        self.k_minimals = {}
        
    
    def sub_lattice(self, bottom_node, top_node):
        
        lattice = defaultdict(dict)
        possible_values = (range(bottom_node[i], top_node[i] + 1) for i in range(len(top_node)))
        
        for node in itertools.product(*possible_values):
            lattice[sum(node)][node] = self.full_lattice[sum(node)][node]

        return lattice
    
    
    def height(self, lattice):
        return len(lattice.keys()) - 1
    
    
    def width(self, lattice, height):
        return len(lattice[height].keys())
    
    
    def policy_parameters(self, node):
        
        params = []
        
        for i in range(len(node)):
            params.append(self.gen_hiers[i][node[i]])
            
        return params
    
    
    def set_k(self, k):
        self.k = k
        
    
    def set_suppression_prop(self, prop):
        self.sup_prop = prop
    
    
    def is_k_anonymous(self, node):
        """
        Returns Boolean indicated whether or not proposed policy
        is k-anonymous, given defined k and suppression percentage thresholds.
        """
        
        prop_notk = self.prop_suppressed(node)
        
        if prop_notk <= self.sup_prop:
            return True
        else:
            return False


    def generalize_attribute(self, col, hier_level, df):
        """
        Helper function for self.generalized_counts, etc.
        Returns:
        Boolean indicating whether or not to include col in groupby_terms.
        Input df with col generalized according to hier_level.
        """

        # generalization type
        gen_type = self.gen_type[col]

        # generalization scheme
        scheme = self.gen_hiers[col][hier_level]

        if gen_type == 'binary':
            # column suppressed
            if not scheme:
                return False, df
            # column not suppressed
            else:
                return True, df
        
        elif gen_type == 'numeric':
            # column suppressed
            if not scheme:
                return False, df
            # column not generalized
            elif scheme == -1:
                return True, df
            # column generalized
            else:
                df[col] = pd.cut(df[col], scheme, right=False).astype(str)
                return True, df

        elif gen_type == 'categorical':
            # column suppressed
            if not scheme:
                return False, df
            # column not generalized
            elif scheme == -1:
                return True, df
            # column generalized
            else:
                for key, value in scheme.items():
                    df.loc[df[col].isin(value), col] = key
                return True, df

        elif gen_type == 'geocode':
            # column suppressed
            if not scheme:
                return False, df
            # column not generalized
            elif scheme == -1:
                return True, df
            # column generalized
            else:
                df[col] = df[col].str[:scheme]
                return True, df


    def generalize_dataset(self, node):
        """
        Generalizes dataset per node's specifications.
        Helper function for self.generalized_counts, etc.
        Returns:
        Generalized dataset
        Groupby terms for generalization counts.
        """

        # original dataset
        temp = self.data.copy()

        # QID terms
        groupby_terms = self.attributes.copy()

        # apply node's generalization policy
        for idx in range(len(node)):

            col = self.attributes[idx]
            hier_level = node[idx]

            keep, temp = self.generalize_attribute(col, hier_level, temp)

            if not keep:
                groupby_terms.remove(col)

        return temp, groupby_terms

    
    def generalized_counts(self, node):
        """
        Generalizes data on QID attributes according to a single policy.
        Returns:
        An array of the class size for each equivalence class.
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset(node)

        # array new equivalence class sizes
        self.equiv_sizes = gen_df.groupby(groupby_terms, sort=False, observed=False)['counts'].sum().values
        
        
    def generalized_counts_distributed(self, node):
        """
        Generalizes data on QID attributes according to a single policy.
        Returns:
        An array of the equivalence class size for each original bin after generalization.
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset(node)

        # get each original bin's new equivalence class size
        self.risk_counts = (gen_df.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('sum')).values
    
    
    def generalized_counts_utility(self, node):
        """
        Generalizes data on QID attributes according to a single policy.
        Returns:
        An array of the redistributed counts over the original bins - used for several intrinsic utility measures.
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset(node)
        
        # redistribute counts
        gen_df['redist'] = (gen_df.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('mean')).values
        
        # create multiplier to zero out suppressed cells
        gen_df['multiplier'] = (gen_df.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('sum')).values
        gen_df['multiplier'] = (gen_df['multiplier'].values >= self.k).astype(int)
        
        self.util_counts = (gen_df['redist'] * gen_df['multiplier']).values


    def generalized_counts_suppressed(self, node):
        """
        Generalizes data on QID attributes according to a single policy.
        Returns:
        An boolean array where True means the records are suppressed - used for PREC calculation
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset(node)
        
        
        # create multiplier to zero out suppressed cells
        equiv_sizes = (gen_df.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('sum')).values
        self.suppressed_idx = (equiv_sizes < self.k)


    def generalized_counts_per_equiv(self, node):
        """
        Prepares data for self.calc_bins_per_equiv
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset(node)
        
        # number of original bins per new equivalence class
        return [(gen_df.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('count')).values,
            (gen_df.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('sum')).values]


    def generalize_dataset_pm(self, node):
        """
        Generalizes dataset per node's specifications - for protective masking.
        Helper function for self.generalized_counts, etc.
        Returns:
        Generalized dataset
        Groupby terms for generalization counts.
        """

        # original dataset
        temp = self.data.copy()

        # QID terms
        groupby_terms = self.attributes.copy()

        # apply node's generalization policy
        for idx in range(len(node)):

            col = self.attributes[idx]
            hier_level = node[idx]

            keep, temp = self.generalize_attribute(col, hier_level, temp)

            if not keep:
                temp[col] = '*'

        return temp, groupby_terms

    
    def transform_dataset_for_protective_masking(self, node):
        """
        Creates a transformed dataset per the specified generalizations.
        Returns:
        self.transformed_dataset
        """

        # generalize QID values
        gen_df, groupby_terms = self.generalize_dataset_pm(node)
        
        # groupby and sum
        gen_df['group_counts'] = gen_df.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('sum').values

        # convert generalized values to string type
        gen_df[self.attributes] = gen_df[self.attributes].astype(str)

        # suppress
        gen_df['suppressed'] = 0
        if self.sup_prop > 0:
            gen_df.loc[gen_df['group_counts'] < self.k, self.attributes] = '*'
            gen_df.loc[gen_df['group_counts'] < self.k, 'suppressed'] = 1
            gen_df['group_counts'] = gen_df.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('sum').values

        # equivalence class number
        gen_df['group'] = gen_df.groupby(groupby_terms, sort=False, observed=False).ngroup()

        # combine transformed values with original values
        gen_df.columns = ["gen_" + x for x in gen_df.columns]
        self.transformed_dataset = pd.concat([self.data, gen_df.drop(columns='gen_counts')], axis=1)
    
    
    def is_tagged_k_anonymous(self, node):
        if self.full_lattice[sum(node)][node] == True:
            return True
        else:
            return False
        
        
    def is_tagged_not_k_anonymous(self, node):
        if self.full_lattice[sum(node)][node] == False:
            return True
        else:
            return False
        
    
    def get_tag(self, node):
        return self.full_lattice[sum(node)][node]
    
    
    def is_tagged(self, node):
        return self.full_lattice[sum(node)][node] != None
    
    
    def tag_k_anonymous(self, node):
        self.full_lattice[sum(node)][node] = True
        self.predictive_k_anonymous(node)
        
        
    def tag_not_k_anonymous(self, node):
        self.full_lattice[sum(node)][node] = False
        self.predictive_not_k_anonymous(node)
        
        
    def predictive_not_k_anonymous(self, node):
        """
        Tags nodes that are connected to and below node as not k-anonymous.
        """
        
        lower_nodes = (range(self.bottom_node[i], node[i] + 1) for i in range(len(node)))
        
        for node in itertools.product(*lower_nodes):
            self.full_lattice[sum(node)][node] = False
            
    
    def predictive_k_anonymous(self, node):
        """
        Tags nodes that are connected to and above as k-anonymous.
        """
        
        upper_nodes = (range(node[i], self.top_node[i] + 1) for i in range(len(node)))
        
        for node in itertools.product(*upper_nodes):
            self.full_lattice[sum(node)][node] = True
            
            
    def clean_up(self, node):
        """
        Removes all nodes in self.k_minimals that are generalizations of node.
        """
        
        keep_node = True
        
        current_level = sum(node)
        
        for level, old_nodes in self.k_minimals.items():
            
            # remove more generalized nodes of current node
            if level > current_level:
                
                for old_node in old_nodes:
                    if np.all(np.array(node) <= np.array(old_node)):
                        self.k_minimals[level].remove(old_node)
            
            #if current node is more generalized than old nodes, do not add current node
            elif level <= current_level:
                
                for old_node in old_nodes:
                    if np.all(np.array(node) >= np.array(old_node)):
                        #print('Keep = False')
                        keep_node = False
                        break
                        
        if keep_node:
            if current_level in self.k_minimals.keys():
                if node not in self.k_minimals[current_level]:
                    self.k_minimals[current_level].append(node)
            else:
                self.k_minimals[current_level] = [node]
                
        #print(self.k_minimals)
        #print()
            
    
    def Kmin(self, bottom_node, top_node):

        L = self.sub_lattice(bottom_node, top_node)
        H = self.height(L)

        #print(L)
        #print()
        
        if not self.done:
            if H > 1:

                h = round(H/2)

                for node in L[list(L.keys())[h]]:
                    #print(node)

                    if self.is_tagged_k_anonymous(node):
                        self.Kmin(bottom_node, node)

                    elif self.is_tagged_not_k_anonymous(node):
                        self.Kmin(node, top_node)

                    elif self.is_k_anonymous(node):
                        self.tag_k_anonymous(node)
                        self.Kmin(bottom_node, node)

                    else:
                        self.tag_not_k_anonymous(node)
                        self.Kmin(node, top_node)

            else:

                if self.is_tagged_not_k_anonymous(bottom_node):
                    N = top_node
                elif self.is_k_anonymous(bottom_node):
                    self.tag_k_anonymous(bottom_node)
                    N = bottom_node
                    if bottom_node == self.bottom_node:
                        self.done=True
                else:
                    self.tag_not_k_anonymous(bottom_node)
                    N = top_node

                #print(N, bottom_node, top_node)
                self.clean_up(N)


    def calc_util_func(self, func, node):

        if func == 'js_div':
            return self.calc_js_div()

        if func == 'query_loss':
            return self.calc_query_loss()

        if func == 'pct_mov':
            return self.calc_abs_pct_moved()

        if func == 'nonu_ent':
            return self.calc_nonuniform_entropy()

        if func == 'dm*':
            return self.calc_DM(node)

        if func == 'num_per_equiv':
            return self.calc_bins_per_equiv(node)

        if func == 'prec':
            return self.calc_PREC(node)
    
    
    def calc_utility(self, node):

        # dataframe of results
        util_vals = pd.DataFrame(columns = self.util_functions, index=[0])
        
        # get bin sizes after generalization
        self.generalized_counts_utility(node)

        for func in self.util_functions:

            util_vals[func] = self.calc_util_func(func, node)
        
        return util_vals

    
    def prop_suppressed(self, node):
        
        # get equivalence class sizes per policy parameters
        self.generalized_counts(node)

        # check if proportion of population not k-anonymous meets suppression percentage
        prop_notk = (self.equiv_sizes[self.equiv_sizes < self.k]).sum() / self.tot_pop

        return prop_notk
    
    
    def calc_js_div(self):
        
        # define distributions
        Q = (self.util_counts + self.prior)
        Q /= Q.sum()
        
        M = (self.P + Q) / 2
        
        return 0.5 * (sum(self.P * np.log2(self.P / M)) + 
                  sum(Q * np.log2(Q / M)))


    def calc_nonuniform_entropy(self):
        """
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records in the dataset. This has the caveat that records who have their
        quasi-identifier suppressed, but not their remaining values, have the same utility
        as those records that are suppressed completely from the dataset.
        """

        # replace suppressed values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts < self.k] = self.tot_pop + 1

        return (-np.log2(self.data.counts.values / util_counts) * self.data.counts.values).sum() / self.tot_pop


    def calc_DM(self, node):
        """
        Same assumption as for calc_nonuniform_entropy.
        Calculates DM* as defined in El Emam's OLA paper.
        """

        # replace suppressed cells with equivalence class size of tot_pop + 1
        util_counts = self.equiv_sizes.copy()
        suppressed = util_counts < self.k
        part1 = (util_counts[~suppressed] ** 2).sum()
        part2 = (util_counts[suppressed] * (self.tot_pop + 1)).sum()
        return (part1 + part2) / self.tot_pop


    def calc_bins_per_equiv(self, node):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        # get bin sizes after generalization
        bin_sizes = self.gen_counts_per_equiv(node)

        return (bin_sizes * self.data.counts.values).sum() / self.tot_pop
    
    
    def calc_query_loss(self):
        return ((np.abs(self.data.counts.values - self.util_counts) /
                 self.data.counts.values) * 100).mean()
    
    
    def calc_abs_pct_moved(self):
        return ((np.abs(self.data.counts.values - self.util_counts).sum()) /
                self.tot_pop) * 100
    
    
    def calc_overall_marketer(self):
        """
        Calculates the overall marketer risk for the given policy.
        """

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts < self.k] = self.tot_pop
        
        mark_risk = (np.nan_to_num(self.data.counts.values / risk_counts).sum() / self.tot_pop)
        
        return mark_risk


    def calc_initial_marketer(self):
        """
        Calculates the marketer risk of the raw dataset.
        """
        return len(self.data[self.data.counts > 0]) / self.tot_pop


    def calc_PREC(self, node):
        """
        Calculates Sweeney's PREC measure.
        """

        # indices of suppressed records
        self.generalized_counts_suppressed(node)

        # create equivalence class level PREC calculation
        df = self.data.copy()
        gen_attrs = ['gen_' + attr for attr in self.attributes]
        df[gen_attrs] = np.array(node)

        # adjust for DGH imbalances
        if self.adjust_prec_dict:

            for adjust_val in self.adjust_prec_dict.keys():
                
                for level, vals in self.adjust_prec_dict[adjust_val].items():
                    
                    if df.loc[0, 'gen_' + adjust_val] == level:

                        df.loc[df[adjust_val].isin(vals), 'gen_' + adjust_val] = 0

        # maximum height of each generalization hierarchy
        max_height = np.array(self.top_node) + 1 # +1 accounts for suppression

        # sum fractions for all records
        df['sum_fracs'] = (df[gen_attrs] / max_height).sum(axis=1)

        # sum fractions for suppressed records
        df.loc[self.suppressed_idx, 'sum_fracs'] = len(self.attributes)

        # normalize sum_fracs
        df['sum_fracs_norm'] = df['sum_fracs'] / len(self.attributes)

        # overall prec
        overall_prec = 1 - (df['sum_fracs_norm'] * df['counts']).sum() / self.tot_pop

        return overall_prec
        
                
    def process_kmin(self):
        """
        Calculate overall utility for k minimal nodes.
        The results can identify the optimal generalization
        according to the selected utility measure.
        """

        self.ans = pd.DataFrame(columns = self.attributes)

        pk_vals = []
        marketer = []
        util_vals = []

        i = 0

        for key, nodes in self.k_minimals.items():

            for node in nodes:
                
                # store generalization levels
                self.ans.loc[i, :] = list(node)
                
                # PK value
                pk_vals.append(self.prop_suppressed(node))

                # marketer risk values
                self.generalized_counts_distributed(node)
                marketer.append(self.calc_overall_marketer())
                
                # utility values
                util_vals.append(self.calc_utility(node))                
                
                i += 1

        self.ans['k'] = self.k
        self.ans['suppression_prop_threshold'] = self.sup_prop
        self.ans['prop_suppressed'] = pk_vals
        self.ans['initial_marketer'] = self.calc_initial_marketer()
        self.ans['marketer'] = marketer
        self.ans['marketer_ratio'] = self.ans['marketer'] / self.ans['initial_marketer']
        self.ans[self.util_functions] = pd.concat(util_vals).values

        if self.util_functions[0] == 'prec':
            self.ans = self.ans.sort_values(self.util_functions, ascending=False).reset_index(drop=True)
        else:
            self.ans = self.ans.sort_values(self.util_functions, ascending=True).reset_index(drop=True)

        if (self.ans['prop_suppressed'] > self.ans['suppression_prop_threshold']).sum() > 0:
            raise ValueError('No generalization meets k and suppression threshold.')


    def find_kmin(self):

        self.Kmin(self.bottom_node, self.top_node)
        
        
    def add_names(self):
        """
        Add policy name column.
        """
        
        self.ans['policy'] = (self.ans[self.attributes]
                              .replace(self.name_hier)
                              .astype(str)
                              .agg(''.join, axis=1))


    def full_transform_not_pm(self):
        print('Finding best nodes ------------------')
        self.find_kmin()
        print('Evaluating best nodes ------------------')
        self.process_kmin()
        print('Adding policy names ------------------')
        self.add_names()
        best_node = self.ans.loc[0, self.attributes].values
        print('Best node:', best_node)
        print('Done ----------------------')


    def full_transform(self):
        print('Finding best nodes ------------------')
        self.find_kmin()
        print('Evaluating best nodes ------------------')
        self.process_kmin()
        print('Adding policy names ------------------')
        self.add_names()
        best_node = self.ans.loc[0, self.attributes].values
        print('Best node:', best_node)
        print('Transforming dataset per optimal policy -------------------')
        self.transform_dataset_for_protective_masking(best_node)
        print('Done ----------------------')


    def calc_group_util_shortcut(self, node):
        
        # initialize dataframes
        group_vars = list(self.data[self.group_var].squeeze().unique())

        if 'prec' in self.util_functions:
            self.group_prec = pd.DataFrame(index = group_vars)
        if 'nonu_ent' in self.util_functions:
            self.group_nonu_ent = pd.DataFrame(index = group_vars)

        self.group_prop_suppr = pd.DataFrame(index = group_vars)
        
        # get policy name
        self.policy_name = self.get_policy_name(node)

        # calculate utility measures
        self.calc_group_utility(node)
        #self.calc_util_inequality()
        
        
    def calc_group_measures(self):
        
        # initialize dataframes
        self.init_group_measures()
        
        for _, policy in self.ans.iterrows():

            # node
            node = tuple(policy[self.attributes].values)

            # k value
            self.k = int(policy['k'])
            # print(self.k)
        
            # get policy name
            self.policy_name = self.get_policy_name(node)

            # calculate marketer risk measures
            self.generalized_counts_distributed(node)
            self.calc_group_marketer()

            # calculate proportion suppressed
            self.calc_group_prop_suppressed()

            # calculate utility measures
            self.calc_group_utility(node)
        
        # calculate inequality
        self.calc_marketer_inequality()
        self.calc_util_inequality()


    def calc_group_utility(self, node):

        self.generalized_counts_utility(node)

        for func in self.util_functions:

            self.calc_group_util_func(func, node)


    def calc_group_util_func(self, func, node):

        if func == 'js_div':
            self.calc_group_js_div()

        if func == 'query_loss':
            self.calc_group_query_loss()

        if func == 'pct_mov':
            self.calc_group_abs_pct_moved()

        if func == 'nonu_ent':
            self.calc_group_nonuniform_entropy(node)

        if func == 'dm*':
            self.calc_group_DM()

        if func == 'num_per_equiv':
            self.calc_group_bins_per_equiv(node)

        if func == 'prec':
            self.calc_group_PREC(node)
        

    def get_policy_name(self, node):
        
        name = ''
        for i in range(len(node)):
            
            attr = self.attributes[i]
            attr_level = node[i]
            
            if attr in self.name_hier.keys():
                name += self.name_hier[attr][attr_level]
            else:
                name += '*'
                
        return name
        
        
    def init_group_measures(self):
        
        group_vars = list(self.data[self.group_var].squeeze().unique())
        
        self.group_marketer = pd.DataFrame(index = group_vars)
        self.group_js = pd.DataFrame(index = group_vars)
        self.group_query_loss = pd.DataFrame(index = group_vars)
        self.group_pct_moved = pd.DataFrame(index = group_vars)
        self.group_prop_suppr = pd.DataFrame(index = group_vars)
        self.group_nonu_ent = pd.DataFrame(index = group_vars)
        self.group_dm_star = pd.DataFrame(index = group_vars)
        self.group_num_per_equiv = pd.DataFrame(index = group_vars)
        self.group_prec = pd.DataFrame(index = group_vars)
        
        
    def calc_group_marketer(self):

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts < self.k] = self.tot_pop
        
        # marketer risk fractions
        mark_df = (pd.concat([self.data[self.group_var],
            pd.DataFrame(data = np.nan_to_num(self.data.counts.values /risk_counts))],
            axis = 1)
        .reset_index(drop=True))
        
        # sum fractions by group
        group_mark_fracs = mark_df.groupby(self.group_var).sum()
        
        # save group-specific marketer risk
        self.group_marketer = (self.group_marketer.merge(
                                           pd.DataFrame({self.policy_name : 
                                                         (np.nan_to_num(group_mark_fracs.values /
                                                                        self.group_counts.values)
                                                            .ravel())},
                                                        index = self.group_counts.index),
                                           left_index=True,
                                           right_index=True,
                                           how='left')).fillna(0)


    def gen_counts_per_equiv(self, node):

        # get bin sizes after generalization
        bin_sizes, risk_counts = self.generalized_counts_per_equiv(node)

        # change suppressed cells number of equivalence classes to number classes suppres
        bin_sizes[risk_counts < self.k] = sum(risk_counts < self.k)
        return bin_sizes
        

    def calc_group_js_div(self):
        """
        Group-specific JS divergence.
        """
        
        # define group_specific probability distributions
        Q_totals = (pd.concat([self.data[self.group_var],
                               pd.DataFrame(data = (self.util_counts + self.prior))],
                               axis = 1)
                      .reset_index(drop=True)
                      .groupby(self.group_var, sort=False)
                      .transform('sum')).values.squeeze()
        
        Q = np.divide((self.util_counts + self.prior), Q_totals)
        M = (self.group_P + Q) / 2
        
        # JS divergence
        js_div = 0.5 * ((self.group_P * np.log2(self.group_P / M)) + 
                  (Q * np.log2(Q / M)))
        
        # put in dataframe
        js_df = pd.DataFrame({self.policy_name : js_div})
        
        # add group columns
        js_df[self.group_var] = self.data[self.group_var]
        
        # divergence values by group
        self.group_js = self.group_js.merge(js_df.groupby(self.group_var).agg({self.policy_name:'sum'}),
                                            left_index = True,
                                            right_index = True,
                                            how = 'left')
        
    
    def calc_group_query_loss(self):
        """
        Mimics the execution of all COUNT queries for each quasi-identifier combination
        at the most specific values. Evalutes the average percent difference across all queries,
        broken down by pre-specified group.
        """

        # construct dataframe
        u_df = self.data[self.group_var + ['counts']].copy()
        u_df['util_counts'] = self.util_counts
        u_df['abs_pct_diff'] = (np.abs(u_df.counts.values - u_df.util_counts.values) / u_df.counts.values) * 100

        # utility loss values by group
        group_loss = u_df.groupby(self.group_var).agg({'abs_pct_diff':'mean'})
        group_loss.columns = [self.policy_name]

        self.group_query_loss = self.group_query_loss.merge(group_loss,
                                                            left_index=True,
                                                            right_index=True,
                                                            how='left')
        
    
    def calc_group_abs_pct_moved(self):
        """
        Mimics the execution of all COUNT queries for each quasi-identifier combination
        at the most specific values. Calculates the cumulative absolute difference across
        all queries, and then divides by the size of the population. Does this for each
        pre-specified group.

        Essentially indicates the percent of the population into which anonymization has induced enough
        uncertainty to "move" them to another bin.
        """

        # construct dataframe
        u_df = self.data[self.group_var + ['counts']].copy()
        u_df['util_counts'] = self.util_counts
        u_df['abs_diff'] = (np.abs(u_df.counts.values - u_df.util_counts.values) / u_df.counts.values)

        # calculate group specific values
        grouped = u_df.groupby(self.group_var).agg(cum_diff = ('abs_diff', 'sum'),
                                                   total_pop = ('counts', 'sum'))

        grouped[self.policy_name] = (grouped.cum_diff.values / grouped.total_pop.values) * 100

        # utility loss values by group
        self.group_pct_moved = self.group_pct_moved.merge(grouped[self.policy_name],
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')
        
        
    def calc_group_prop_suppressed(self, node):
        """
        Calculates percent of group whose records are suppressed under the policy.
        """

        self.generalized_counts_distributed(node)
        
        # calculate
        df = (self.data[self.group_var + ['counts']]).copy()
        df['num_suppr'] = df.counts * (self.risk_counts < self.k) # mark suppressed
        group_suppr = (df.groupby(self.group_var).agg({'num_suppr':'sum', 'counts':'sum'}))
        group_suppr['prop_suppressed'] = group_suppr['num_suppr'] / group_suppr['counts']

        # store results
        self.group_prop_suppr = self.group_prop_suppr.merge(group_suppr[['prop_suppressed']],
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')


    def calc_group_nonuniform_entropy(self, node):
        """
        Calculates group-specific non-uniform entropy.
        """
        self.generalized_counts_distributed(node)
        # replace zero values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts  < self.k] = self.tot_pop + 1

        # sum and groupby group variable
        df = self.data[self.group_var + ['counts']].copy()
        df['entropy'] = -np.log2(df.counts.values / util_counts) * df.counts.values
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'entropy':'sum'})
        grouped[self.policy_name] = grouped['entropy'] / grouped['counts']

        # store results
        self.group_nonu_ent = self.group_nonu_ent.merge(grouped[[self.policy_name]],
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')


    def calc_group_PREC(self, node):
        """
        Calculates group-specific PREC.
        """

        # indices of suppressed records
        self.generalized_counts_suppressed(node)

        # create equivalence class level PREC calculation
        df = self.data.copy()
        gen_attrs = ['gen_' + attr for attr in self.attributes]
        df[gen_attrs] = np.array(node)

        # adjust for DGH imbalances
        if self.adjust_prec_dict:

            for adjust_val in self.adjust_prec_dict.keys():
                
                for level, vals in self.adjust_prec_dict[adjust_val].items():
                    
                    if df.loc[0, 'gen_' + adjust_val] == level:
        
                        df.loc[df[adjust_val].isin(vals), 'gen_' + adjust_val] = 0

        # maximum height of each generalization hierarchy
        max_height = np.array(self.top_node) + 1  # +1 accounts for suppression

        # sum fractions for all records
        df['sum_fracs'] = (df[gen_attrs] / max_height).sum(axis=1)

        # sum fractions for suppressed records
        df.loc[self.suppressed_idx, 'sum_fracs'] = len(self.attributes)

        # normalize sum_fracs
        df['sum_fracs_norm'] = (df['sum_fracs'] / len(self.attributes)) * df['counts']

        # final prec result
        group_prec = df.groupby(self.group_var).agg({'sum_fracs_norm':'sum',
            'counts':'sum'})
        group_prec[self.policy_name] = group_prec['sum_fracs_norm'] / group_prec['counts']

        # store results
        self.group_prec = self.group_prec.merge(1 - group_prec[self.policy_name],
                                                left_index=True,
                                                right_index=True,
                                                how='left')


    def calc_group_bins_per_equiv(self, node):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        # get bin sizes after generalization
        bin_sizes = self.gen_counts_per_equiv(node)

        # sum by group_var
        df = self.data[self.group_var + ['counts']].copy()
        df['group_bins'] = (bin_sizes * df.counts.values)
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'group_bins':'sum'})
        grouped[self.policy_name] = grouped['group_bins'] / grouped['counts']

        # store results
        self.group_num_per_equiv = self.group_num_per_equiv.merge(grouped[[self.policy_name]],
                                                left_index=True,
                                                right_index=True,
                                                how='left')


    def calc_group_DM(self):
        """
        Groups-specific DM*
        """

        # replace suppressed values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts  < self.k] = self.tot_pop + 1

        # sum and groupby group variable
        df = self.data[self.group_var + ['counts']].copy()
        df['equiv_class_mult'] = util_counts * df.counts.values
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'equiv_class_mult':'sum'})
        grouped[self.policy_name] = grouped['equiv_class_mult'] / grouped['counts']

        # store results
        self.group_dm_star = self.group_dm_star.merge(grouped[[self.policy_name]],
                                                left_index=True,
                                                right_index=True,
                                                how='left')
        

    def calc_marketer_inequality(self):

        gini_coeffs = np.apply_along_axis(gini, 0, self.group_marketer.values)

        self.ans['marketer_inequality'] = gini_coeffs


    def calc_util_inequality_func(self, func):

        if func == 'js_div':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_js.values)
            self.ans['js_div_inequality'] = gini_coeffs

        if func == 'query_loss':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_query_loss.values)
            self.ans['query_loss_inequality'] = gini_coeffs

        if func == 'pct_mov':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_pct_moved.values)
            self.ans['pct_mov_inequality'] = gini_coeffs

        if func == 'nonu_ent':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_nonu_ent.values)
            self.ans['nonu_ent_inequality'] = gini_coeffs

        if func == 'dm*':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_dm_star.values)
            self.ans['dm*_inequality'] = gini_coeffs

        if func == 'num_per_equiv':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_num_per_equiv.values)
            self.ans['num_per_equiv_inequality'] = gini_coeffs

        if func == 'prec':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_prec.values)
            self.ans['prec_inequality'] = gini_coeffs
        
    
    def calc_util_inequality(self):

        # utility functions
        for func in self.util_functions:

            self.calc_util_inequality_func(func)
        
        # prop suppressed
        gini_coeffs_ps = np.apply_along_axis(gini, 0, self.group_prop_suppr.values)
        self.ans['prop_suppr_inequality'] = gini_coeffs_ps
            


class OLA_row_level:

    def __init__(self, df, attributes, generalization_hierarchies, generalization_types, k, suppression_prop, name_hier,
                  group_var, util_func='nonu_ent', ignore_cols=[], adjust_prec_dict=False):
        
        self.df = df[attributes].copy()
        self.attributes = attributes.copy()
        self.gen_hiers = generalization_hierarchies.copy()
        self.gen_type = generalization_types.copy()
        self.k = k
        self.sup_prop = suppression_prop
        self.name_hier = name_hier.copy()
        self.group_var = group_var
        self.ignore_cols = ignore_cols.copy()
        self.ignore_cols += ['orig_group']
        self.util_func = util_func
        self.adjust_prec_dict = adjust_prec_dict
        

    def condense_to_hist(self):

        # keep track of original group
        self.df['orig_group'] = self.df.groupby(self.attributes, sort=False, observed=False).ngroup()
        
        # histogram version of row-level dataset
        self.df_hist = (self.df.groupby(self.attributes, sort=False, observed=False)
                        .agg(counts=(self.attributes[0], 'count'),
                             orig_group=('orig_group', 'first'))
                        .reset_index())
    
    def find_best_transformation(self):
        
        # OLA_hist obj
        self.obj = OLA_histogram(data=self.df_hist,
                                attributes=self.attributes,
                                generalization_hierarchies=self.gen_hiers,
                                generalization_types = self.gen_type,
                                k=self.k,
                                suppression_prop=self.sup_prop,
                                name_hier=self.name_hier,
                                group_var=self.group_var,
                                ignore_cols=self.ignore_cols,
                                util_functions=[self.util_func],
                                adjust_prec_dict=self.adjust_prec_dict)
        
        # execute
        self.obj.full_transform()
        
        # clean up
        self.obj.transformed_dataset.drop(['gen_orig_group'], axis=1, inplace=True)
        
    
    def transform_row_level_dataset(self):
        
        # columns from transformed dataset to keep
        keep_cols = ['orig_group', 'gen_suppressed', 'gen_group', 'gen_group_counts', 'counts']
        keep_attr = ['gen_' + x for x in self.attributes]
        keep_cols += keep_attr
        
        # merge original dataset with transformed values
        self.transformed_row = (pd.merge(self.df.reset_index(col_fill='index'),
                                         self.obj.transformed_dataset[keep_cols],
                                         how = 'left',
                                         on = 'orig_group')
                                .drop('orig_group', axis=1)
                                .set_index('index'))
        self.transformed_row.index.name = ''
        
        # store policy name
        self.transformed_row_policy = self.obj.ans.loc[0, 'policy']
        
        # store policy node values
        self.transformed_row_policy_levels = self.obj.ans.loc[0, self.attributes]


    def run(self):

        self.condense_to_hist()
        self.find_best_transformation()
        self.transform_row_level_dataset()


    def transform_external_dataset_per_policy(self, df):
        """
        Used to transform X_test in ML workflow.
        """

        df = df.copy()

        node = self.transformed_row_policy_levels.values

        # apply node's generalization policy
        for idx in range(len(node)):

            col = self.attributes[idx]
            hier_level = node[idx]

            keep, df = self.obj.generalize_attribute(col, hier_level, df)

            if not keep:
                df[col] = '*'

        # rename columns
        cols = []
        for col in df.columns:
            if col in self.attributes:
                cols.append('gen_' + col)
            else:
                cols.append(col)

        df.columns = cols

        return df


