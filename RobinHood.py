"""
Code to implement protective masking (k-anonymity/k-concealment - based)
anonymization approach.
"""

import numpy as np
import pandas as pd


### Helper functions

def calc_D(A, target_k, C):
    return np.ceil(A * (target_k - A) / C)


### Sample from histogram class

class sample_from_histogram:
    
    
    def __init__(self, data, rng):
        
        self.df = data
        self.rng = rng
        
        
    def one_sample(self, population_size, num_samples, replacement):
        
        return self.rng.choice(population_size, num_samples, replace=replacement)
            
            
    def process_samples(self, col_idx, samps, counts_field):
        
        samps_df = pd.DataFrame({counts_field:samps+0.5,
                                'sample':int(1)})
        #samps_df['sample'] = int(1)
        
        cc = self.df[counts_field].cumsum().to_frame().reset_index(names='bin')

        concat_df = (pd.concat([cc,
                               samps_df],
                               axis=0)
                       .sort_values([counts_field]))

        concat_df['bin'] = concat_df['bin'].bfill().astype(int)

        ppl_per_bin = concat_df.loc[concat_df['sample'] == 1, 'bin'].value_counts()
        
        self.samples[ppl_per_bin.index, col_idx] += ppl_per_bin.values
        
    
    def get_cumulative_counts(self, counts_field):
        
        self.cumulative_counts = self.df[counts_field].cumsum().to_frame().reset_index(names='bin')
        
    
    def initialize_samples(self, n_sims):
        
        self.samples = np.zeros([len(self.df), n_sims])
            
    
    def get_samples(self, pct, n_sims, counts_field, replacement=False):
        
        """
        Less memory intensive. Can run on laptop. May be slower
        than other version when greater memory resources are available.
        """
        
        # cumulative counts for argmin
        self.get_cumulative_counts(counts_field)
        
        # total population size
        total_pop = self.cumulative_counts[counts_field].values[-1]
        
        # number of samples
        num_samples = round(pct * total_pop)
                        
        # initialize answer
        self.initialize_samples(n_sims)
        
        for i in range(n_sims):
            
            # random sample
            samps = self.one_sample(total_pop, num_samples, replacement)
            
            # process samples
            self.process_samples(i, samps)
            
    
    def get_samples_counts(self, num_samples, n_sims, counts_field, replacement=False):
        
        """
        Less memory intensive. Can run on laptop. May be slower
        than other version when greater memory resources are available.
        """
        
        # cumulative counts for argmin
        self.get_cumulative_counts(counts_field)
        
        # total population size
        total_pop = self.cumulative_counts[counts_field].values[-1]
                        
        # initialize answer
        self.initialize_samples(n_sims)
        
        for i in range(n_sims):
            
            # random sample
            samps = self.one_sample(total_pop, int(num_samples), replacement)
            
            # process samples
            self.process_samples(i, samps, counts_field)


### ProtectiveMasking classes

class ProtectiveMasking_histogram:

	def __init__(self, df, target_k, group_var, gen_attributes, C_thresh, D_thresh,
	gen_group_counts_field = 'gen_group_counts', gen_group_field = 'gen_group', 
	original_counts_field = 'counts', random_seed=False):

		self.df = df.copy()
		self.target_k = target_k
		self.group_var = group_var
		self.attributes = gen_attributes
		self.C_thresh = C_thresh
		self.D_thresh = D_thresh
		self.counts_field = gen_group_counts_field
		self.group_field = gen_group_field
		self.orig_counts_field = original_counts_field

		if random_seed:
			self.rng = np.random.default_rng(seed = random_seed)
		else:
			self.rng = np.random.default_rng()

		self.other_attr = self.attributes.copy()
		self.other_attr.remove(group_var)

		self.initialize_answer()


	def initialize_answer(self):

		self.ans = self.df.copy()

		# assign new group number based on other_qid
		self.ans['other_attr'] = self.ans.groupby(self.other_attr, sort=False, observed=False).ngroup()

		# num_masked/unmasked column
		self.ans['n_masked'] = 0
		self.ans['n_unmasked'] = self.ans[self.orig_counts_field].copy()

		# C and D columns
		self.ans['C'] = 0
		self.ans['D'] = 0

		# generalize flag
		self.ans['Needs_generalization'] = 0

		# effective k
		self.ans['effective_k'] = self.ans[self.counts_field].copy().astype(float)


	def groups_below_target(self):

		self.below_target = (
			self.ans[self.ans[self.counts_field] < self.target_k]
			.groupby(self.group_field)
			.agg({'effective_k':'first'})
			.sort_values('effective_k')
			.index.values
			)


	def find_masking_groups(self, group):

		# other_attr number for group
		masking_group_num = self.ans.loc[self.ans[self.group_field] == group, 'other_attr'].values[0]

		# masking group
		self.masking_group = (
			(self.ans[self.ans.other_attr == masking_group_num]
			.groupby(self.group_field)
            .agg({self.counts_field:'first'})))


	def meets_target_k(self, group):

		if self.ans.loc[self.ans[self.group_field] == group, 'effective_k'].values[0] >= self.target_k:
			return True
		else:
			return False


	def calc_num_to_mask(self, group):

		# check if already meets threshold
		if self.meets_target_k(group):
			return 0, 0, False

		# inital num records masked within target group
		C = self.ans[self.ans[self.group_field] == group]['n_masked'].sum()

		# intial num records masked outside target group
		D_groups = self.masking_group[self.masking_group.index != group].index
		D = self.ans[self.ans[self.group_field].isin(D_groups)]['n_masked'].sum()

		# total number of records in target group
		tot_C = self.masking_group.loc[group, self.counts_field]

		# total number of records outside target (and other self.below_target groups)
		tot_sample = self.masking_group[~self.masking_group.index.isin(self.below_target)].sum().values[0]

		# if none have been masked already, change to 1
		if C == 0:
		    mask_C = 1
		    C = 1
		else:
		    mask_C = 0

		# target group's original size    
		A = tot_C

		# get D for initial C value
		mask_D = calc_D(A, self.target_k, C) - D

		# need to generalize initially set to zero
		generalize = False

		# if D is already met
		if mask_D <= 0:
			if generalize:
				print('Group:', group)
				print('Generalize:', generalize)
				print()

		# iterate to determine C, D, and whether generalization is needed
		# in the event that initial C is too small
		while (D + max(mask_D, 0)) > (tot_sample * self.D_thresh):
		    
		    C += 1
		    mask_C += 1
		    
		    mask_D = calc_D(A, self.target_k, C) - D
		    
		    if C >= (tot_C * self.C_thresh):
		        generalize=True
		        break
		
		if generalize:
			print('Group:', group)
			print('Generalize:', generalize)
			print()

		return mask_C, mask_D, generalize


	def mask_from_D_groups(self, group, num_mask):

		# get D group index values
		# this D group only includes groups outside self.below_target
		D_groups = self.masking_group[~self.masking_group.index.isin(self.below_target)].index

		# init sample object
		obj = sample_from_histogram(
			self.ans[self.ans[self.group_field].isin(D_groups)]['n_unmasked'].reset_index(),
			self.rng)

		# randomly assign masking
		obj.get_samples_counts(num_samples = num_mask, n_sims = 1, counts_field = 'n_unmasked')

		# set new unmasked values
		self.ans.loc[self.ans[self.group_field].isin(D_groups), 'n_masked'] += obj.samples[:, 0]


	def mask_from_C_group(self, group, num_mask):

		# init sample object
		obj = sample_from_histogram(
			self.ans[self.ans[self.group_field] == group]['n_unmasked'].reset_index(),
			self.rng)

		# randomly assign masking
		obj.get_samples_counts(num_samples = num_mask, n_sims = 1, counts_field = 'n_unmasked')

		# set new unmasked values
		self.ans.loc[self.ans[self.group_field] == group, 'n_masked'] = obj.samples[:, 0]


	def recalc_unmasked(self):

		self.ans['n_unmasked'] = self.ans[self.orig_counts_field] - self.ans['n_masked']


	def update_C_and_D(self):

		# indices of masking group
		change = self.ans[self.group_field].isin(self.masking_group.index)

		# total masked in masking group
		tot_mask = self.ans[change]['n_masked'].sum()

		# update C
		self.ans.loc[change, 'C'] = self.ans.loc[change].groupby(self.group_field)['n_masked'].transform('sum')

		# update D
		self.ans.loc[change, 'D'] = tot_mask - self.ans.loc[change, 'C']


	def update_effective_k(self):

		# indices of masking group
		change = self.ans[self.group_field].isin(self.masking_group.index)

		# update effective_k
		self.ans.loc[change, 'effective_k'] = (self.ans.loc[change, self.counts_field] 
                                + (self.ans.loc[change, 'C'] * self.ans.loc[change, 'D']) 
                                / self.ans.loc[change, self.counts_field])


	def run(self):


		self.groups_below_target()

		for group in self.below_target:

			self.find_masking_groups(group)

			mask_C, mask_D, generalize = self.calc_num_to_mask(group)
			
			if generalize:
				print()
				print('GENERALIZE!!')
				print()
				self.ans.loc[self.ans[self.group_field] == group, 'Needs_generalization'] = 1
			else:
				if mask_C > 0:
					self.mask_from_C_group(group, mask_C)
				if mask_D > 0:
					self.mask_from_D_groups(group, mask_D)

			self.recalc_unmasked()
			self.update_C_and_D()
			self.update_effective_k()



class ProtectiveMasking_row:

	def __init__(self, df, target_k, group_var, gen_attributes, C_thresh, D_thresh,
	gen_group_counts_field = 'gen_group_counts', gen_group_field = 'gen_group', 
	original_counts_field = 'counts', random_seed=False):

		self.df = df.copy()
		self.target_k = target_k
		self.group_var = group_var
		self.attributes = gen_attributes
		self.C_thresh = C_thresh
		self.D_thresh = D_thresh
		self.counts_field = gen_group_counts_field
		self.group_field = gen_group_field
		self.orig_counts_field = original_counts_field

		self.num_mask = 0

		if random_seed:
			self.rng = np.random.default_rng(seed = random_seed)
		else:
			self.rng = np.random.default_rng()

		self.other_attr = self.attributes.copy()
		self.other_attr.remove(group_var)

		self.initialize_answer()


	def initialize_answer(self):

		self.ans = self.df.copy()

		# assign new group number based on other_qid
		self.ans['other_attr'] = self.ans.groupby(self.other_attr, sort=False, observed=False).ngroup()

		# num_masked/unmasked column
		self.ans['masked'] = 0
		# self.ans['n_masked'] = 0
		# self.ans['n_unmasked'] = self.ans[self.orig_counts_field].copy()

		# C and D columns
		self.ans['C'] = 0
		self.ans['D'] = 0

		# generalize flag
		self.ans['Needs_generalization'] = 0

		# effective k
		self.ans['effective_k'] = self.ans[self.counts_field].copy().astype(float)


	def groups_below_target(self):

		self.below_target = (
			self.ans[self.ans[self.counts_field] < self.target_k]
			.groupby(self.group_field)
			.agg({'effective_k':'first'})
			.sort_values('effective_k')
			.index.values
			)


	def find_masking_groups(self, group):

		# other_attr number for group
		masking_group_num = self.ans.loc[self.ans[self.group_field] == group, 'other_attr'].values[0]

		# masking group
		self.masking_group = (
			(self.ans[self.ans.other_attr == masking_group_num]
			.groupby(self.group_field)
            .agg({self.counts_field:'first'})))


	def meets_target_k(self, group):

		if self.ans.loc[self.ans[self.group_field] == group, 'effective_k'].values[0] >= self.target_k:
			return True
		else:
			return False


	def calc_num_to_mask(self, group):

		# check if already meets threshold
		if self.meets_target_k(group):
			return 0, 0, False

		# inital num records masked within target group
		C = self.ans[self.ans[self.group_field] == group]['masked'].sum()

		# intial num records masked outside target group
		D_groups = self.masking_group[self.masking_group.index != group].index
		D = self.ans[self.ans[self.group_field].isin(D_groups)]['masked'].sum()

		# total number of records in target group
		tot_C = self.masking_group.loc[group, self.counts_field]

		# total number of records outside target (and other self.below_target groups)
		tot_sample = self.masking_group[~self.masking_group.index.isin(self.below_target)].sum().values[0]

		# if none have been masked already, change to 1
		if C == 0:
		    mask_C = 1
		    C = 1
		else:
		    mask_C = 0

		# target group's original size    
		A = tot_C

		# get D for initial C value
		mask_D = calc_D(A, self.target_k, C) - D

		# need to generalize initially set to zero
		generalize = False

		# if D is already met
		if mask_D <= 0:
			if generalize:
				print('Group:', group)
				print('Generalize:', generalize)
				print()

		# iterate to determine C, D, and whether generalization is needed
		# in the event that initial C is too small
		while (D + max(mask_D, 0)) > (tot_sample * self.D_thresh):
		    
		    C += 1
		    mask_C += 1
		    
		    mask_D = calc_D(A, self.target_k, C) - D
		    
		    if C >= (tot_C * self.C_thresh):
		        generalize=True
		        break
		
		# print('Group:', group)		        
		# print('Sample C:', mask_C)
		# print('Sample D:', mask_D)
		if generalize:
			print('Group:', group)
			print('Generalize:', generalize)
			print()

		return mask_C, mask_D, generalize


	def mask_from_D_groups(self, group, num_mask):

		self.num_mask += num_mask

		# get D group index values
		# this D group only includes groups outside self.below_target
		D_groups = self.masking_group[~self.masking_group.index.isin(self.below_target)].index

		# find records to sample from for masking
		to_sample = self.ans[(self.ans[self.group_field].isin(D_groups)) & (self.ans['masked'] == 0)].index

		# randomly choose which records to mask
		to_mask = self.rng.choice(to_sample, int(num_mask), replace=False)

		# set new masked values
		self.ans.loc[to_mask, 'masked'] = 1


	def mask_from_C_group(self, group, num_mask):
		
		self.num_mask += num_mask

		# find records to sample from for masking
		to_sample = self.ans[(self.ans[self.group_field] == group) & (self.ans['masked'] == 0)].index

		# randomly choose which records to mask
		to_mask = self.rng.choice(to_sample, int(num_mask), replace=False)

		# set new masked values
		self.ans.loc[to_mask, 'masked'] = 1


	# def recalc_unmasked(self):

	# 	self.ans['n_masked'] = self.ans.groupby(self.ans[self.group_field])['masked'].transform('sum').value_counts()

	# 	self.ans['n_unmasked'] = self.ans[self.orig_counts_field] - self.ans['n_masked']


	def update_C_and_D(self):

		# indices of masking group
		change = self.ans[self.group_field].isin(self.masking_group.index)

		# total masked in masking group
		tot_mask = self.ans[change]['masked'].sum()

		# update C
		self.ans.loc[change, 'C'] = self.ans.loc[change].groupby(self.group_field)['masked'].transform('sum')

		# update D
		self.ans.loc[change, 'D'] = tot_mask - self.ans.loc[change, 'C']


	def update_effective_k(self):

		# indices of masking group
		change = self.ans[self.group_field].isin(self.masking_group.index)

		# update effective_k
		self.ans.loc[change, 'effective_k'] = (self.ans.loc[change, self.counts_field] 
                                + (self.ans.loc[change, 'C'] * self.ans.loc[change, 'D']) 
                                / self.ans.loc[change, self.counts_field])


	def transform_datset(self):

		self.ans.loc[self.ans['masked'] == 1, self.group_var] = 'masked'


	def run(self):


		self.groups_below_target()

		for group in self.below_target:

			self.find_masking_groups(group)

			mask_C, mask_D, generalize = self.calc_num_to_mask(group)
			
			if generalize:
				print()
				print('GENERALIZE!!')
				print()
				self.ans.loc[self.ans[self.group_field] == group, 'Needs_generalization'] = 1
			else:
				if mask_C > 0:
					self.mask_from_C_group(group, mask_C)
				if mask_D > 0:
					self.mask_from_D_groups(group, mask_D)

			#self.recalc_unmasked()
			self.update_C_and_D()
			self.update_effective_k()
			self.transform_datset()

	def calc_PREC(self, init_obj, adjust_dict = False):

		# initialize dataframe
		df = self.ans[init_obj.attributes + ['gen_suppressed', 'masked']].copy()

		# set k-init DGH levels
		df[self.attributes] = init_obj.transformed_row_policy_levels.values

		# adjust if needed
		if adjust_dict:

		    for adjust_val in adjust_dict.keys():

		        for level, vals in adjust_dict[adjust_val].items():

		            if df['gen_' + adjust_val].values[0] == level:

		                df.loc[df[adjust_val].isin(vals), 'gen_' + adjust_val] = 0

		# maximum height of each generalization hierarchy
		max_height = np.array(init_obj.obj.top_node) + 1  # +1 accounts for suppression

		# change level for masked records
		max_h = max_height[np.where(np.array(init_obj.attributes) == self.group_var[4:])[0][0]]
		df.loc[df.masked == 1, self.group_var] = max_h

		# sum fractions for all records
		df['sum_fracs'] = (df[self.attributes] / max_height).sum(axis=1)

		# sum fractions for suppressed records
		df.loc[df.gen_suppressed == 1, 'sum_fracs'] = len(self.attributes)

		# normalize sum_fracs
		df['sum_fracs_norm'] = df['sum_fracs'] / len(self.attributes)

		# overall prec
		overall_prec = 1 - df['sum_fracs_norm'].sum() / len(df)

		return overall_prec



	def calc_group_PREC(self, init_obj, adjust_dict = False):

		# initialize dataframe
		df = self.ans[init_obj.attributes + ['gen_suppressed', 'masked']].copy()

		# set k-init DGH levels
		df[self.attributes] = init_obj.transformed_row_policy_levels.values

		# adjust if needed
		if adjust_dict:

		    for adjust_val in adjust_dict.keys():

		        for level, vals in adjust_dict[adjust_val].items():

		            if df['gen_' + adjust_val].values[0] == level:

		                df.loc[df[adjust_val].isin(vals), 'gen_' + adjust_val] = 0

		# maximum height of each generalization hierarchy
		max_height = np.array(init_obj.obj.top_node) + 1  # +1 accounts for suppression

		# change level for masked records
		max_h = max_height[np.where(np.array(init_obj.attributes) == self.group_var[4:])[0][0]]
		df.loc[df.masked == 1, self.group_var] = max_h

		# sum fractions for all records
		df['sum_fracs'] = (df[self.attributes] / max_height).sum(axis=1)

		# sum fractions for suppressed records
		df.loc[df.gen_suppressed == 1, 'sum_fracs'] = len(self.attributes)

		# normalize sum_fracs
		df['sum_fracs_norm'] = df['sum_fracs'] / len(self.attributes)

		# group
		group_prec = df.groupby(self.group_var[4:]).agg({'sum_fracs_norm':'sum',
			'masked':'count'})

		group_prec['PREC'] = 1 - group_prec['sum_fracs_norm'] / group_prec['masked']

		return group_prec[['PREC']]


	def calc_nonuniform_entropy(self, obj):
		"""
		Assumes that the probability of suppressed records = # records for the bin / 
		total # records + 1 in the dataset. Also assumes probability of a record in 
		a masked group is equal to 1 / size of masked group.
		"""

		# results
		df = self.ans.copy()

		# number original equivalence class that is masked
		df['orig_masked'] = (df.groupby(obj.attributes)
                               ['masked']
                               .transform('sum').values)

		# number original equivalence class that is not masked
		df['orig_unmasked'] = df['counts'] - df['orig_masked']

		# numerator of entropy equation
		df['num'] = df['orig_unmasked']
		df.loc[df['masked'] == 1, 'num'] = df['orig_masked']

		# denominator of entropy equation
		df['denom'] = df.groupby(self.attributes + ['masked']).transform('size').values

		# change group size for suppressed records
		df.loc[df['gen_suppressed'] == 1, 'denom'] = len(df) + 1

		# entropy
		df['entropy'] = -np.log2(df['num'] / df['denom'])

		return df['entropy'].mean()


	def calc_group_nonuniform_entropy(self, obj):
		"""
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records + 1 in the dataset. Also assumes probability of a record in 
        a masked group is equal to 1 / size of masked group.
        """

		# results
		df = self.ans.copy()

		# number original equivalence class that is masked
		df['orig_masked'] = (df.groupby(obj.attributes)
                               ['masked']
                               .transform('sum').values)

		# number original equivalenc class that is not masked
		df['orig_unmasked'] = df['counts'] - df['orig_masked']

		# numerator of entropy equation
		df['num'] = df['orig_unmasked']
		df.loc[df['masked'] == 1, 'num'] = df['orig_masked']

		# denominator of entropy equation
		df['denom'] = df.groupby(self.attributes + ['masked']).transform('size').values

		# change group size for suppressed records
		df.loc[df['gen_suppressed'] == 1, 'denom'] = len(df) + 1

		# entropy
		df['entropy'] = -np.log2(df['num'] / df['denom'])

		return df.groupby(self.group_var[4:]).agg({'entropy':'mean'})


	def calc_group_masking_proportions(self):

		df = self.ans.groupby('race').agg({'masked':'sum', 'counts':'count'})
		df['masking_prop'] = df['masked'] / df['counts']
		return df[['masking_prop']]



