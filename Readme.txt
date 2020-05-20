Investigate the effects of species richness on ND and FD
Python code corresponding to:
Species richness increases fitness differences, but does not affect niche differences
https://doi.org/10.1101/823070

The files cluster_full_factorial_NFD_computation.py, analyse_cluster.py,



Computer code:
	
	cluster_full_factorial_NFD_computation.py
		Computes the full factorial virtual experiment
		Generates the files `NFD_val/NFD_values *.npz`
		Should be called on a cluster with the jobarray_flullfactorial.sh file:
			sbatch --array=1-432 jobarray_flullfactorial.sh
	analyse_cluster.py
		Combines the files `NFD_val/NFD_values *.npz` into the file `fullfactorial_data.csv`
	interaction_estimation.py
		Computes distributions for realistic interaction strength
		Generates figure S3
	LV_real_multispec_com.py
		Generates all subcommunities from the literature review and computes ND and FD
		as well as other coexistence related properties for all subcommunities
		Generates NFD_real_LV.csv
	higher_order_models.py
		Computes NFD for a given community model with higher order interactions
	LV_multi_functions
		Compute NFD for a LV multispeciescommunity
		Computes for which communities NFD can be computed
	regrssion_fullfactorial.py
		computes the linear regressions for the fullfactorial design
		generates the file regression_fullfactorial.csv
	fullfactorial_effects.py
		Computes which factors have a significant effect on ND or FD
		creates fullfactorial_slope_percentiles.csv
		creates fullfactorial_significance.csv
	literature_data_overview.py
		Creates the table S1
	
	plot*
		generates the corresponding plot *.pdf
	
Data files
	
	fullfactorial_data.csv
		Contains results from fullcatorial simulations
		ord1, ord2, ord3, con, cor, ord1_strenght and indirect contain the factor settings
		case and id summarizes all factor settings
		a is the average interspecific interaction strength per community
		ND and FD report the ND and FD of species 0
	NFD_val/NFD_values *.npz
		Contains all simulation results from the simulations with * as factor levels
	fullfactorial_significance.csv
		p-values for each factor (rows) and each response variable (columns)
	fullfactorial_slope_precentiles.csv
		This is table S1
	literature_data_overview.csv	
		This is table S2
	LV_multispec.csv
		The original interaction matrices from the literature with references
	NFD_real_LV.csv
		NFD for literature data and references
		
	
	
		