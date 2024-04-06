"""
   funcs for refine spectra

"""
import tools
import itertools
import numpy as np

def flat_repo_list(rep_lists):
    return [rep for rep_list in rep_lists for rep in rep_list]

def get_large_error_reps(dataset, threshold=1.5):

    replicates_list = []
    for pep in dataset.peptides:
        for tp in pep.timepoints:
            replicates_list.append(tp.replicates)


    error_lsit = []
    large_error_list = []
    for rep_list in replicates_list:
        if len(rep_list) == 1:
            continue
        if len(set([rep.timepoint.time for rep in rep_list])) != 1:
            continue    
        rep_combinations = list(itertools.combinations(rep_list, 2))
        abs_error =  np.average([tools.get_sum_ae(com[0].isotope_envelope, com[1].isotope_envelope) for com in rep_combinations])
        error_lsit.append(abs_error)
        if abs_error > threshold:
            #idf = get_identifer(rep_list[0].peptide)
            large_error_list.append(rep_list)
            #print(idf, abs_error)

    #flat_large_error_list = [rep for rep_list in large_error_list for rep in rep_list]

    return large_error_list



def find_low_intensity_reps(replicates_list, threshold=10):

    low_intens_reps = []
    for rep_list in replicates_list:
        max_intenstiy_rep = max(rep_list, key=lambda rep: rep.raw_ms['Intensity'].max())
        for rep in rep_list:
            intens_ratio = max_intenstiy_rep.raw_ms['Intensity'].max() / rep.raw_ms['Intensity'].max()
            if intens_ratio > threshold:
                #print(intens_ratio)
                low_intens_reps.append(rep)
    
    return low_intens_reps


def refine_large_error_reps(dataset):

    while True:
        flat_large_error_list = flat_repo_list(get_large_error_reps(dataset, threshold=1.5))

        for rep in flat_large_error_list:
            try:
                tools.filter_by_thoe_ms(rep)
            except:
                print(rep.peptide.sequence, rep.timepoint.time,)

        flat_large_error_list = flat_repo_list(get_large_error_reps(dataset, threshold=1.5))
        if flat_large_error_list == []:
            break
        tools.remove_reps_from_dataset(flat_large_error_list, dataset)

        flat_large_error_list = get_large_error_reps(dataset, threshold=1.2)
        low_intens_reps = find_low_intensity_reps(flat_large_error_list, threshold=5)
        tools.remove_reps_from_dataset(low_intens_reps, dataset)


    


    
