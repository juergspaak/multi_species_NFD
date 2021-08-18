Investigate the effects of species richness on ND and FD

Python code corresponding to:
Species richness increases fitness differences, but does not affect niche differences
https://doi.org/10.1101/823070

Computer code:
	analyse_cluster.py
		Combines the files `NFD_val/NFD_values *.npz` into the file `fullfactorial_data.csv`
		The NFD_val/NFD_values *.npz files are not part of the repository, these must first be created
		with running cluster_full_factorial_NFD_computation.py (potentially using jobarray_fullfactorial.sh
	cluster_full_factorial_NFD_computation.py
		Computes the full factorial virtual experiment
		Generates the files `NFD_val/NFD_values *.npz`
		Should be called on a cluster with the jobarray_flullfactorial.sh file:
			sbatch --array=1-432 jobarray_flullfactorial.sh
	jobarray_fullfactorial.sh
		sbatch script to run on the cluster utilized
	higher_order_models.py
		Computes NFD for a given community model with higher order interactions
	interaction_estimation.py
		Computes distributions for realistic interaction strength
	LV_multi_functions
		Compute NFD for a LV multispecies community
		Computes for which communities NFD can be computed
	LV_real_multispec_com.py
		Generates all subcommunities from the literature review and computes ND and FD
		as well as other coexistence related properties for all subcommunities
		Generates NFD_real_LV.csv
	regression_fullfactorial.py
		computes the linear regressions for the fullfactorial design
		generates the file regression_fullfactorial.csv	
	Table*.py
		Creates the corresponding table	
	plot*
		generates the corresponding figure *.pdf
	
Data files
	Table_*.csv	
		File for corresponding table in Latex file	
	regression_fullfactorial.csv
		Contains the linear regressions of niche and fitness differences for the fullfactorial design
	LV_multispec.csv
		The original interaction matrices from the literature with references
	evolution_LV.csv
		Contains the end points of the evolution of LV dynamics as used in figure 5
		Columns: species, identity of the species within a community;
			m, peak of resource consumption
			w, standard deviation of resource consumption
			Sinit, species richness of community
			intr_growth, intrinsic growth rate
			alphas, species interaction
	fullfactorial_data.csv
		Contains results from fullcatorial simulations
		ord1, ord2, ord3, con, cor, ord1_strenght and indirect contain the factor settings
		case and id summarizes all factor settings
		a is the average interspecific interaction strength per community
		ND and FD report the ND and FD of species 0
		
	
	
		