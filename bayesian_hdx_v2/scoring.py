"""
   Scoring functions for HDX models
"""
from __future__ import print_function
import numpy
from bayesian_hdx_v2 import tools
import scipy
import scipy.stats
import math
import numpy as np
from numba import jit, prange, njit
import numba

# numba.set_num_threads(2)

class ScoringFunction(object):
    '''
    A scoring function consists of a forward model + noise model along with
    priors for each parameter within the model and the data.

    It depends on a model representation (from bayesian_hdx::model -- should be remaned Representation)
    a Forward model, a Noise model and sets of priors for each type of parameter

    '''
    def __init__(self, fwd_model):
        '''
        Input is a model, which must have a function get_model(), which returns the current model
        self.representation 
        '''
        self.forward_model = fwd_model
        self.priors = []

    def add_prior(self, prior):
        self.priors.append(prior)


    def evaluate(self, model_values=None, peptides=None):
        '''
        Evaluates an entire model based on a single set of model values.
        '''
        total_score = 0

        if model_values is None:
            model_values = self.representation.get_current_model()

        total_score = self.forward_model.evaluate(model_values, peptides)
        #print(" FM:",total_score, model_values)
        for p in self.priors:
            #print("  ",p.evaluate(model_values))
            total_score += p.evaluate(model_values)

        #print("   ", model_values[2:-1])
        return total_score



class ProtectionFactorNaturalAbundancePrior(object):
    '''
    A prior on the likelihood of observing a given protection factor given no other information

    Can choose among built-in functions or supply your own function.

    Functions are evaluated by numpy.interp()
    '''
    def __init__(self, prior_type, input_bins=None, input_pfs=None, gaussian_parameters=[(1,0.8,3), (5,1.6,80)]):
        '''
        prior_type can be "bmrb", "knowledge", "user" or "uninformative"
        '''
        self.bins = []
        self.probs = []
        self.prior_type = prior_type
        if prior_type == "bmrb":
            for i in sorted(bmrb_pf_histograms.keys()):
                self.bins.append(i)
                self.probs.append(bmrb_pf_histograms[i])
        elif prior_type == "knowledge":
            for i in sorted(knowledge_pf_histograms.keys()):
                self.bins.append(i)
                self.probs.append(knowledge_pf_histograms[i])   
        elif prior_type == "uninformative":
            for i in range(-2,14):
                self.bins.append(i)
                self.probs.append(1.0) 
        elif prior_type == "Gaussian":
            self.gaussians = []
            for g in gaussian_parameters:
                self.gaussians.append((scipy.stats.norm(g[0], g[1]),g[2]))

        elif prior_type == "user":
            if input_bins is not None and input_pfs is not None:
                if len(input_bins) == len(input_pfs):
                    self.bins = input_bins
                    self.probs = input_pfs
                else:
                    raise Exception("scoring.SingleProtectionFacotrPrior.set_prior: Length of bins and probabilities is not the same")
            else:
                raise Exception("scoring.SingleProtectionFacotrPrior.set_prior: Must supply input_bins and input_pfs for a user-supplied prior")
        # make sure we normalize!
        self.probs = numpy.linalg.norm(self.probs)

    def evaluate_pf_likelihood(self, pf):
        return numpy.interp(pf, self.bins, self.probs)

    def evaluate(self, model):
        score = 0
        if self.prior_type == "Gaussian":
            for m in model:
                prob = 1
                for g in self.gaussians:
                    prob *= g[0].pdf(m)*g[1]
                    #print(g[0].pdf(m), g[1], m, prob)
                score += -1*numpy.log(prob +0.0000000001)
        else:
            for m in model:
                score += -1*numpy.log(self.evaluate_pf_likelihood(m))
        return score


class SigmaPriorCauchy(object):
    '''
    A prior that is applied to the sigma for each timepoint or data
    '''
    def __init__(self, state, sigma_estimate=1.0, prior_scale=20.0):
        self.state = state
        self.scale = prior_scale

    def evaluate(self):
        # Prior on the sigma value. Long tailed to allow for outlier values.
        sigma_prior_score = 0
        for d in self.state.data:
            for p in d.get_peptides():
                for tp in p.get_timepoints():
                    sigma_prior_score += (1 / tp.sigma**2) * math.exp(-tp.sigma0**2 / tp.sigma**2)

        return sigma_prior_score * self.prior_scale


class LogSigmaPriorNormal(object):
    '''
    A prior that is applied to the log of the sigma.
    Keeps the value of sigma positive and 
    '''
    def __init__(self, state, log_sigma_estimate=0.0, sd=1.0, prior_scale=1.0):

        self.state = state
        self.prior_scale = prior_scale
        self.point_estimate = log_sigma_estimate
        self.sd = sd
        self.distribution = scipy.stats.norm(log_sigma_estimate, sd)

    def evaluate(self, model):
        # Prior on the sigma value. Long tailed to allow for outlier values.
        sigma_prior_score = 0
        for d in self.state.data:
            for p in d.get_peptides():
                for tp in p.get_timepoints():
                    sigma_prior_score += -1*numpy.log(self.distribution.pdf(numpy.log(tp.sigma)))

        return sigma_prior_score * self.prior_scale


class FlatDistribution(object):
    '''
    Simple object to return a flat prior.

    Compatible with scipy.stats distributions that contain a pdf() function

    Returns the same value regardless of input.
    '''
    def __init__(self, scale=0.1):
        self.scale = scale

    def pdf(self, value):
        return self.scale


class ResiduePfPrior(object):
    '''
    Given a set of estimates for the protection factor of each residue, apply a 
    Gaussian prior on that residue.

    Ideally determined from an MD simulation or, perhaps, NMR data of certain residues
    '''
    def __init__(self, pf_estimates, scale=1.0):
        '''
        pf_estimates is in the form (mean, sd).
        For residues without an estimate, denoted by an SD < 0, the prior is flat for any value
        '''
        self.prior_scale = scale
        self.priors = []
        for p in range(len(pf_estimates)):
            prior = pf_estimates[p]
            if prior[1] < 0:
                self.priors.append(FlatDistribution())
            else:
                self.priors.append(scipy.stats.norm(prior[0], prior[1]))

    def evaluate(self, model):
        if len(model) != len(self.priors):
            raise Exception("ERROR scoring.ResiduePfPrior: The length of the Pf prior is not th same as the model")

        score = 0
        for p in range(len(model)):
            #print(p, model[p], -1*numpy.log(self.priors[p].pdf(model[p])))
            score += -1*numpy.log(self.priors[p].pdf(model[p]))

        return score * self.prior_scale


class ExtremeValuePrior(object):
    '''
    Given a array of extreme value parameters, apply a prior on the protection factor of each residue.

    '''
    def __init__(self, pf_categories, scale=1.0):
        '''
        -1: low exchaning residue, high protection factor
        0: normal exchanging residue
        1: high exchanging residue, low protection factor
        '''
        self.prior_scale = scale
        self.priors = []
        for p in range(len(pf_categories)):
            prior = pf_categories[p]
            if prior == 0:
                self.priors.append(FlatDistribution())
            elif prior == 1:
                self.priors.append(scipy.stats.norm(1.5, 2.0))
            elif prior == -1:
                self.priors.append(scipy.stats.norm(6, 4.0))

    def evaluate(self, model):
        if len(model) != len(self.priors):
            raise Exception("ERROR scoring.ResiduePfPrior: The length of the Pf prior is not th same as the model")

        score = 0
        for p in range(len(model)):
            #print(p, model[p], -1*numpy.log(self.priors[p].pdf(model[p])))
            score += -1*numpy.log(self.priors[p].pdf(model[p]))
        return score * self.prior_scale


class GaussianNoiseModel(object):
    '''
    This is a Forward Model.  Actually a Forward Model plus noise model.
    It converts a model (set of protection factors) into an expected value for each piece of data 
    and then derives a likelihood for the data:model pair using the noise model.

    This model gathers its standard deviation parameters from the individual timepoint objects.
    '''
    def __init__(self, state, truncated=False, bounds=(None, None)):
        self.truncated=truncated

        self.state = state

        if truncated:
            # 10 and -10 are numerically equivalent to no truncation factor when
            # data is in the range of 0-->1
            if bounds[1] is None:
                self.upper_bound = 10
            else:
                self.upper_bound = bounds[1]

            if bounds[0] is None:
                self.lower_bound = -10
            else:
                self.lower_bound = bounds[0]

    def replicate_score(self, model, exp, sigma):
        # Forward model
        #priors = self.model_prior(protection_factor) * self.exp_prior(exp) * self.sigma_prior()
        raw_likelihood = math.exp(-((model-exp)**2)/(2*sigma**2))/(sigma*math.sqrt(2*numpy.pi))
        if self.truncated:
            raw_likelihood *= 1/ ( 0.5 * ( scipy.special.erf( (self.upper_bound-exp)/sigma * math.sqrt(3.1415) ) - scipy.special.erf( (self.lower_bound-exp)/sigma * math.sqrt(3.1415) ) )  )
        return raw_likelihood

    def peptide_confidence_score(self, peptide):
        # User-definable function for converting peptide confidence into a likelihood.
        # The function must be evaluatable between 0 and 1. 
        pass 


    def calculate_peptides_score(self, peptides, protection_factors):
        '''
        Will deprecate calculate_dataset_score. Given a list of peptides,
        calculate the score. Useful for calculating changes that only affect
        a susbset of peptides.
        '''
        self.state.calculate_residue_incorporation(protection_factors)
        
        total_score = 0
        if peptides is None:
            # peptides are the ones that we recalculate
            peptides = self.state.get_all_peptides()
            non_peptides = []
        else:
            # non_peptides are the ones where we simply read the score from last time (didn't change)
            non_peptides = list(set(self.state.get_all_peptides())-set(peptides))
        #print("PEP:", len(peptides), len(non_peptides))
        for pep in peptides:
            peptide_score = 0

            d = pep.get_dataset()

            observable_residues = pep.get_observable_residue_numbers()

            # Cycle over all timepoints
            for tp in pep.get_timepoints():
                # initialize tp score to the sigma prior
                tp_score = 0
                #model_tp_raw_deut = self.sum_incorporations(self.calculate_residue_incorporation(self.output_model.model_protection_factors)[d], observable_residues, tp.time)
                model_tp_raw_deut = 0

                #try:
                #    self.state.residue_incorporations[d][0][tp.time]
                #except:


                for r in observable_residues:
                    model_tp_raw_deut += self.state.residue_incorporations[d][r][tp.time]

                #------------------------------
                # Here is where we would add back exchange estimate
                #------------------------------

                # Convert raw deuterons into a percent
                model_tp_deut = float(model_tp_raw_deut)/pep.num_observable_amides * 100

                # Calculate a score for each replicate
                for rep in tp.get_replicates():
                    replicate_likelihood = self.replicate_score(model=model_tp_deut, exp=rep.deut, sigma=0.5) 
                    if replicate_likelihood <= 0:
                        rep.set_score(10000000000)
                    else:
                        rep.set_score(-1*math.log(replicate_likelihood))

                    tp_score += rep.get_score()

                # Set the timepoint score
                tp.set_score(tp_score)

                peptide_score += tp_score

            total_score += peptide_score
            #print(pep.sequence, peptide_score, protection_factors)

        for pep in non_peptides:
            #print(pep.sequence, pep.get_score(), protection_factors)
            total_score += pep.get_score()

        self.total_score = total_score
        return total_score

    def evaluate(self, model, peptides):
        return self.calculate_peptides_score(peptides, model)


class GaussianNoiseModelIsotope(object):
    '''
    This is a Forward Model.  Actually a Forward Model plus noise model.
    It converts a model (set of protection factors) into an expected value for each piece of data 
    and then derives a likelihood for the data:model pair using the noise model.

    This model gathers its standard deviation parameters from the individual timepoint objects.
    '''
    def __init__(self, state, envelope_sigma=0.3, centroid_sigma=0.5, w_envelope = 1.0, w_centroid = 1.0, truncated=False, bounds=(None, None)):
        
        self.truncated = truncated
        self.state = state
        self.envelope_sigma = envelope_sigma
        self.centroid_sigma = centroid_sigma

        self.w_envelope = w_envelope
        self.w_centroid = w_centroid

        if truncated:
            self.upper_bound = bounds[1] if bounds[1] is not None else 10
            self.lower_bound = bounds[0] if bounds[0] is not None else -10
        else:
            self.upper_bound = None
            self.lower_bound = None


    def replicate_score(self, model=None, exp=None, model_centroid=None, exp_centroid=None):

        sum_ae = np.abs(model - exp).sum(axis=1)

        envelope_likelihood = np.exp(-(sum_ae ** 2) / (2 * self.envelope_sigma ** 2)) / (self.envelope_sigma * np.sqrt(2 * np.pi))
        
        centroid_likelihood = np.exp(-((model_centroid - exp_centroid) ** 2) / (2 * self.centroid_sigma ** 2)) / (self.centroid_sigma * np.sqrt(2 * np.pi))
    
        
        raw_likelihood = (envelope_likelihood)**self.w_envelope * (centroid_likelihood)**self.w_centroid

        
        if self.truncated:
            upper_gaussian = (self.upper_bound - exp_centroid) / sigma
            lower_gaussian = (self.lower_bound - exp_centroid) / sigma
            gaussian_truncation_factor = 1 / (0.5 * (erf(upper_gaussian / np.sqrt(2)) - erf(lower_gaussian / np.sqrt(2))))

            raw_likelihood *= gaussian_truncation_factor

        # set 10000000000 when raw_likelihood <0 
        raw_likelihood[raw_likelihood <= 0] = 10000000000

        total_score = np.sum(-1*np.log(raw_likelihood))        
        
        return total_score
    

    def peptide_confidence_score(self, peptide):
        # User-definable function for converting peptide confidence into a likelihood.
        # The function must be evaluatable between 0 and 1. 
        pass 


    def calculate_peptides_score(self, peptides, protection_factors):
        '''
        Will deprecate calculate_dataset_score. Given a list of peptides,
        calculate the score. Useful for calculating changes that only affect
        a susbset of peptides.
        '''
        self.state.calculate_residue_incorporation(protection_factors)
        
        total_score = 0
        if peptides is None:
            # peptides are the ones that we recalculate
            peptides = self.state.get_all_peptides()
            non_peptides = []
        else:
            # non_peptides are the ones where we simply read the score from last time (didn't change)
            non_peptides = list(set(self.state.get_all_peptides())-set(peptides))
        #print("PEP:", len(peptides), len(non_peptides))
        
        all_rep_data = self.state.all_rep_data
        all_rep_data['residue_incorporations'] = self._prepare_residue_incorporations_data(peptides)
        all_rep_data['model_centroid'], all_rep_data['model_full_iso'] = calculate_model_full_iso(all_rep_data)

        total_score += self.replicate_score(model=all_rep_data['model_full_iso'],
                                            exp=all_rep_data['isotope_envelope'],
                                            model_centroid=all_rep_data['model_centroid'],
                                            exp_centroid=all_rep_data['exp_centroid'],)

        for pep in non_peptides:
            #print(pep.sequence, pep.get_score(), protection_factors)
            total_score += pep.get_score()

        self.total_score = total_score
        #print(total_score)
        return total_score


    def _prepare_residue_incorporations_data(self, peptides):
        
        d = self.state.data[0]

        res_incorp = []

        for pep in peptides:

            observable_residues = pep.get_observable_residue_numbers()

            for tp in [tp for tp in pep.get_timepoints() if tp.time != 0]:

                model_tp_raw_deut = []
                for r in observable_residues:
                    model_tp_raw_deut.append(self.state.residue_incorporations[d][r][tp.time])
                
                model_tp_raw_deut = tools.custom_pad(numpy.array(model_tp_raw_deut), 20, 0.0)

                for rep in tp.get_replicates(): 
                
                    res_incorp.append(model_tp_raw_deut)
        
        return np.array(res_incorp).reshape(-1, 20)


    def evaluate(self, model, peptides):
        return self.calculate_peptides_score(peptides, model)


@jit(nopython=True)
def erf(x):
    # Approximation of the error function
    # Constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y



@njit(parallel=True)
def convolve_and_truncate(p_D_array, t0_p_D_array):
    num_samples = len(p_D_array)
    truncated_len = 20
    results = np.empty((num_samples, truncated_len), dtype=np.float32)  # 提前定义所需的输出数组大小

    for i in prange(num_samples):
        conv_result = np.convolve(p_D_array[i], t0_p_D_array[i])
        results[i] = conv_result[:truncated_len]  # 截断卷积结果

    return results


@njit(parallel=True)
def compute_event_probabilities(deuterations):
    n = len(deuterations)
    results = np.empty((n, 20), dtype=np.float32)
    for i in prange(n):
        # Assuming event_probabilities expects an array slice and returns a float
        # Process a slice of the first 19 elements for each array in deuterations
        result = tools.event_probabilities(deuterations[i][:19])
        results[i] = result  # Storing a float into the results array
    return results



def calculate_model_full_iso(all_rep_data):
    # Calculate raw deuteration levels
    raw_deuteration =all_rep_data['residue_incorporations'] * all_rep_data['max_d'] / all_rep_data['num_observable_amides']
    model_centroid = np.sum(raw_deuteration, axis=1)

    p_D_array = compute_event_probabilities(raw_deuteration)
    
    # Convolve and truncate results
    model_full_iso = convolve_and_truncate(p_D_array, all_rep_data['t0_p_D'])

    return model_centroid, model_full_iso


# def replicate_score(model, exp, sigma):
    
#     sum_ae = np.abs(model - exp).sum(axis=1)
#     raw_likelihood = np.exp(-(sum_ae ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

#     # set 10000000000 when raw_likelihood <0 
#     raw_likelihood[raw_likelihood <= 0] = 10000000000

#     total_score = np.sum(-1*np.log(raw_likelihood))
    
#     return total_score
