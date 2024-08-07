"""@namespace IMP.hdx.data
   Classes that store HX data
"""
from bayesian_hdx_v2 import tools
from bayesian_hdx_v2 import hxio
import numpy
import numpy.random
import pandas as pd
import math
from hdxrate import k_int_from_sequence
import uuid

def calculate_percent_d(dataset):
    # Calculates the averaged %D over all timepoints in a peptide.
    # Reports one value per peptide. Chalmers / Griffin / Pascal method
    # Currently just prints to screen
    for pep in dataset.get_peptides():
        avg_deut = pep.calculate_average_deuteration()
        if avg_deut is not None:
            print(">", pep.start_residue, pep.sequence, avg_deut)
        else:
            print("Peptide" + pep.sequence + "has no data")

def calculate_delta_percent_d(dataset1, dataset2):
    # Simple function takes two datasets and calculates the
    # average percent D difference between each peptide
    pass


class Conditions(object):
    """ Class containing a set of conditions for an HDX experiment """
    def __init__(self, temperature=293, pH=7.0, saturation=1.0):
        self.temperature = temperature
        self.pH = pH
        self.saturation = saturation


class Dataset(object):
    """ Characterizes a single HDX dataset for a macromolecular system.
    Requires a Macromolecule object and can create a state object
    Consists of a list of Peptides with associated timepoints, along with experimental parameters
    Intrinsic rates are calculated and stored here

    Rates are calculated with respect to a maximum theoretical rate (self.max_rate)
    observable at these conditions.  These "protection_factors" 
    """   

    def __init__(self, name, conditions, sequence, input_file=None, error_estimate=1.0, 
                    offset=1, number_fast_exchanging_amides=2, percent_deuterium=False):
        self.raw_data_file = input_file
        self.conditions = conditions
        self.sequence = sequence
        self.peptides = []
        self.nfastamides = number_fast_exchanging_amides # use this to utilize NMR data (set to zero)
        if sequence is not None:
            self.intrinsic = self.calculate_intrinsic_rates()
            # The maximum theoretically observable rate at these conditions is from
            # the third residue in "XCCX". NOT 100% SURE ABOUT THIS!
            self.max_rate = self.get_max_rate()
        self.offset = offset # offset between MS file and FASTAFILE residue numbers
        self.sigma_estimate = error_estimate
        self.name = name
        self.sigma0 = error_estimate
        self.sigma = error_estimate
        self.peptide_dict = {}
        self.percent_deuterium = percent_deuterium
        self.times = set([tp.time for tp in self.get_all_timepoints()])
        self.max_D = None

    def get_max_rate(self):
        # Returns the theoretical maximum rate for these dataset conditions
        try:
            max_rate = self.max_rate
        except:
            max_rate = tools.get_sequence_intrinsic_rates("ACCA", self.conditions.pH, self.conditions.temperature, log=True)[-1]            
            self.max_rate = max_rate
        return max_rate

    def collect_times(self):
        self.times = set([tp.time for tp in self.get_all_timepoints()])

    def get_peptides_with_residue(self, resnum):
        # returns a list of peptide objects that overlap the given residue
        # number
        peptides = []
        for pep in self.get_peptides():
            if resnum > pep.start_residue and resnum < pep.start_residue + len(pep.sequence):
                if resnum in pep.get_observable_residue_numbers():
                    peptides.append(pep)
        return peptides

    def get_all_tp_avg_sigmas(self):
        # Returns a list of the average and sds from each timepoint.
        sigmas = []
        for tp in self.get_all_timepoints():
            sigmas.append(tp.get_avg_sd())
        return sigmas

    def set_tp_sigmas_by_replicate(self, replicate_number, sigma):
        # This function will set a sigma value by the replicate_id attribute 
        # of each replicate object
        pass

    def is_this_peptide_in_dataset(self, sequence, start_residue, charge_state):
        for pep in self.get_peptides():
            if sequence==pep.sequence and start_residue==pep.start_residue and charge_state==pep.charge_state:
                return True, pep
        return False, None

    def peptide_sequence_consistency(self, peptide):
        if self.sequence is None:
            return True
        return tools.subsequence_consistency(self.sequence, peptide.get_sequence(), peptide.start_residue)

    def calculate_pooled_sd(self):
        # Calculate the average standard deviation over all timepoints
        total_tps = 0
        sd = 0
        for p in self.peptides:
            for tp in p.timepoints:
                if tp.calculate_sd > 0:
                    sd += tp.calculate_sd
                    total_tps += 1
        return sd / float(total_tps)

    def calculate_observable_rate_bounds(self, threshold = 0.01):
        # The first and last timepoints define a range of 
        # observable exchange rates.
        # These are calculated as those that are only contribute 1% at 
        # each extreme. (or to a user defined threshold)
        if threshold >= 0.5:
            raise Exception("Dataset.get_observable_rate_bounds: Threshold must be < 0.5 and ideally < 0.1")
        if threshold > 0.1:
            raise Warning("Dataset.get_observable_rate_bounds: This threshold value ("+str(threshold)+") is very high and may result in incorrect modeling.  Unless you are sure you want to do this, try 0.01")

        if len(self.get_all_timepoints()) == 0:
            raise Exception("Dataset.get_observable_rate_bounds: No timepoints for this dataset")

        first_time_point = min([t.time for t in self.get_all_timepoints()])
        last_time_point = max([t.time for t in self.get_all_timepoints()])

        fastest_logk = numpy.log10(-numpy.log(threshold)/first_time_point)
        slowest_logk = numpy.log10(-numpy.log(1-threshold)/last_time_point)

        self.observable_bounds = (slowest_logk-0.5, fastest_logk+0.5)

        return self.observable_bounds

    def calculate_observable_protection_factors(self):
        # Using the observable rate bounds and intrinsic rates
        # calculate the observable protection factors at each residue
        fast = self.observable_bounds[1]
        slow = self.observable_bounds[0]
        observable_pfs = []
        for i in self.intrinsic:
            if i == numpy.inf or i == -1 * numpy.inf:
                observable_pfs.append((numpy.inf, numpy.inf))
            else:
                observable_pfs.append((i-fast, i-slow))

        self.observable_protection_factors = observable_pfs

        return self.observable_protection_factors

    def get_observable_rate_bounds(self):
        return self.observable_bounds

    def get_peptides(self):
        return self.peptides

    def get_peptides_by_id(self, ids):
        peptides = []
        for i in ids:
            peptides.append(peptide_dict[i])
        return peptides

    def ensure_unique_peptide_id(self, peptide, modify_if_not_unique=True):
        '''
        Given a new peptide, make sure the id of this peptide is not already
        in the peptide list
        '''
        id_list = [p.id for p in self.peptides]
        if peptide.id in id_list:
            if modify_if_not_unique:
                peptide.id = str(peptide.id) + "_X"

            else:
                raise Exception("Peptide id" + str(peptide.id) + "is already in list")

    def add_peptide(self, new_peptide):
        '''
        Takes a peptide object and adds it to the State peptide list
        Returns peptide object
        '''
        
        if self.sequence is not None:
            if not self.peptide_sequence_consistency(new_peptide):
                target_pep = self.sequence[new_peptide.start_residue-1:new_peptide.start_residue+len(new_peptide.sequence)-1]
                #print(new_peptide.start_residue)
                print("Peptide ", new_peptide.sequence, new_peptide.start_residue, " does not fit in ", target_pep )
                raise Exception("Exiting at Dataset.add_peptide")
            # If the peptide is already in the dataset, return that peptide
            isin, pep = self.is_this_peptide_in_dataset(new_peptide.sequence, new_peptide.start_residue, new_peptide.charge_state)
            if isin:
                return pep
            else:
                self.peptides.append(new_peptide)
                new_peptide.set_dataset(self)
                self.peptide_dict[new_peptide.get_id()] = new_peptide
                return self.peptides[-1]
        else:
            self.peptides.append(new_peptide)
            new_peptide.set_dataset(self)
            self.peptide_dict[new_peptide.get_id()] = new_peptide
            return self.peptides[-1]

    def get_all_timepoints(self):
        # Returns all timepoint objects contained in the dataset
        tps = []
        for p in self.peptides:
            for tp in p.get_timepoints():
                tps.append(tp)
        return tps

    def get_all_times(self):
        # Returns the set of times (in seconds) contained in this dataset
        return self.times

    def create_peptide(self, sequence, start_residue, peptide_id=None, sigma=None, charge_state=None, retention_time=None):
        '''
        Manually creates a peptide object
        adds it to the end of Dataset peptide list

        Returns the peptide object
        '''
        if sigma is None:
            sigma = self.sigma

        if peptide_id is None:
            peptide_id = f"{start_residue}-{start_residue+len(sequence)-1} {sequence}"
        new_peptide = Peptide(self, sequence, start_residue, peptide_id, 
                        sigma=sigma, 
                        charge_state=charge_state, 
                        retention_time=retention_time)
        return self.add_peptide(new_peptide)

    def get_sequence(self):
        return self.sequence

    def set_state(self, state):
        self.state = state

    def calculate_intrinsic_rates(self):
        #self.intrinsic = tools.get_sequence_intrinsic_rates(self.get_sequence(), self.conditions.pH, self.conditions.temperature, log=True)
        self.intrinsic = numpy.log10(k_int_from_sequence(self.get_sequence(), 
                                             self.conditions.temperature, 
                                             self.conditions.pH,
                                             d_percentage=self.conditions.saturation*100))
        return self.intrinsic

    def get_intrinsic_rates(self):
        return self.intrinsic

    def get_state(self):
        return self.state

    def write_to_file(self, outfile):
        # Converts the dataset into nested dictionaries and dumps them to 
        # a standard json output.
        import json
        dataset = {}

        dataset["name"] = self.name
        dataset["conditions"] = self.conditions.__dict__
        dataset["sequence"] = self.sequence
        dataset["error_estimate"] = self.sigma_estimate
        dataset["raw_data_file"] = self.raw_data_file
        dataset["offset"] = self.offset
        dataset["num_fast_exchanging_amides"] = self.nfastamides
        dataset["percent_deuterium"] = self.percent_deuterium
        dataset["max_rate"] = self.max_rate

        peptides = {}
        for pep in self.get_peptides():
            pep_dict = {}
            #print(pep.sequence)
            pep_dict["sequence"] = pep.sequence
            pep_dict["start_residue"] = pep.start_residue
            pep_dict["charge_state"] = pep.charge_state
            pep_dict["retention_time"] = pep.retention_time
            pep_dict["sigma"] = pep.sigma
            timepoints = {}
            for tp in pep.get_timepoints():
                replicates = {}
                tp_dict = {}
                for rep in tp.get_replicates():
                    rep_dict = {}

                    #to avioid circular references
                    properties_to_exclude = ['timepoint', 'peptide', 'raw_ms', 'isotope_envelope', 'sn_ratio', 'max_d']
                    for key, value in rep.__dict__.items():
                        if key not in properties_to_exclude:
                            rep_dict[key] = value
                
                    #rep_dict["deuteration"] = rep.deut
                    #rep_dict["reliability"] = rep.reliability
                    #rep_dict["retention_time"] = rep.rt
                    replicates[rep.experiment_id] = rep_dict
                tp_dict["time"] = tp.time
                tp_dict["sigma"] = tp.sigma
                tp_dict["replicates"] = replicates
                timepoints[tp.time] = tp_dict

            pep_dict["timepoints"] = timepoints
            peptides[pep.get_id()] = pep_dict

        dataset["peptides"] = peptides

        with open(outfile,"w") as f:
            json.dump(dataset,f)
        self.raw_data_file=outfile

        return dataset

    def get_score(self):
        # loop over all peptides and timepoints and sum up the scores.
        score = 0
        for pep in self.get_peptides():
            score += pep.get_score()
        self.score = score
        return score

    def set_sigma(self, sigma):
        self.sigma = sigma
        for pep in self.get_peptides():
            pep.set_sigma(sigma)

    def sum_residue_incorporations(self, deut_by_residue):
        #print(deut_by_residue)
        for pep in self.get_peptides():
            residues = pep.get_observable_residue_numbers()
            for tp in pep.get_timepoints():
                deut = 0
                for r in residues:
                    #print(pep.sequence, r, tp.time, deut, deut_by_residue[r][tp.time])
                    deut += deut_by_residue[r][tp.time]
                #print("   ", pep.sequence, tp.time, deut)
                tp.set_deuteration(deut * self.conditions.saturation)

    def delete_peptide(self, sequence, start_res, charge_state):
        # Remove a given peptide from this dataset
        for p in range(len(self.get_peptides())):
            if self.peptides[p].sequence == sequence and self.peptides[p].start_residue == start_res and self.peptides[p].charge_state==charge_state:
                self.peptides.remove(self.peptides[p])
                break
            if p == len(self.get_peptides())-1:
                raise Warning("Peptide", sequence, start_res, charge_state, "does not exist in this dataset")


    def simulate_all_data(self, protection_factors, timepoints, replicates, error):
        # Remove all timepoints from each peptide and replace with new simulated data
        # based on the given protection factors

        self.get_intrinsic_rates()

        # clear all peptides
        for p in self.peptides:
            p.timepoints = []

        for p in self.peptides:
            # First, get the rate for each residue in the peptide:
            rates = []
            #print(p.start_residue, p.start_residue+len(p.sequence), p.sequence)
            for i in range(p.start_residue+2,p.start_residue+len(p.sequence)):
                rates.append(self.intrinsic[i-1]-protection_factors[i-1])

            for tp in timepoints:
                timepoint = p.add_timepoint(tp)
                for i in range(replicates):
                    # Calculate deuteration
                    deut=0
                    for r in rates:
                        deut += tools.calculate_deut(r, tp)

                    pdeut = deut / float(p.get_number_of_observable_amides()) * 100
                    pdeut += numpy.random.normal(0, pdeut*error/100)

                    #print(p.start_residue, p.sequence, tp, deut, rates)
                    timepoint.add_replicate(pdeut)

    def get_residue_information_content(self, protection_factors):
        '''
        Calculate for all residues, the amount of information at each residue, for each timepoiint and peptide

        For each peptide, the information at Pf = P for a residue is defined as
        1/(number of observable amides) * Resolving_power(P)

        Where resolving power is the sum over all timepoints, tp, of: 
            10^[(k_ex/P)*exp(-10^(k_ex/P)*tp)
        where k_ex is the intrinsic rate of the amide at the dataset conditions (T, pH)
        '''
        residue_information = numpy.zeros((len(self.sequence), len(protection_factors)))
        for r in range(1,len(self.sequence)+1):
            peptides = self.get_peptides_with_residue(r)
            for p in peptides:
                factor = 1.0/p.get_number_of_observable_amides()
                for tp in p.get_timepoints():
                    t = tp.time
                    rps = numpy.zeros(len(protection_factors))
                    for i in range(len(protection_factors)):
                        pf = protection_factors[i]
                        logk = self.intrinsic[r-1] - pf
                        rate = -1*(10**logk)*t
                        rps[i] = 10**logk*numpy.exp(rate)
                    residue_information[r-1][:]+=rps/numpy.linalg.norm(rps)*factor

        return residue_information





class Peptide(object):
    """ Class that stores a list of Timepoint data for each experimental HDX peptide fragment.
        Each Peptide object is bound to a single Dataset.
        Contains experiment-specific data
        @param sequence - sequence of the peptide
        @param start_residue - starting residue of the peptide
        @param peptide_id - a unique peptide ID
        @param sigma - an error 

    """
    def __init__(self, dataset, sequence, start_residue, peptide_id, charge_state=None, sigma=5.0, retention_time=None):
        self.set_dataset(dataset)
        self.id = peptide_id
        self.sequence = sequence
        self.timepoints = []
        self.start_residue = int(start_residue)
        self.end_residue = int(start_residue) + len(sequence) - 1
        self.num_observable_amides = self.calculate_number_of_observable_amides()
        self.sigma = sigma
        self.charge_state = charge_state
        self.retention_time = retention_time
        self.mass = tools.calculate_mass(self.sequence)
        self.observable_residue_numbers = self.calculate_observable_residue_numbers()

    def calculate_average_deuteration(self):
        # Average the deuteration for all timepoints
        total_deut = 0
        num_observations = 0
        for tp in pep.get_timepoints():
            for rep in tp.get_replicates():
                total_deut += 0
                num_observation += 1
        if num_observations > 0:
            avg_deut = total_deut / float(num_observations)
            return avg_deut
        else:
            return None

    def get_score(self):
        score = 0
        for tp in self.get_timepoints():
            score += tp.get_score()
        return score

    def get_best_charge_state(self):
        tp0_reps = [rep for tp in self.timepoints for rep in tp.replicates if rep.timepoint.time == 0 ]
        # lowest kl divergence 
        best_rep = min(tp0_reps, key=lambda x: x.get_score())
        self.best_charge_state = best_rep.charge_state
        self.best_t0_replicate = best_rep
        return self.best_charge_state
    
    def get_sequence(self):
        return self.sequence

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    def get_residue_numbers(self):
        #return a list of the residue numbers in the peptide
        return range(self.start_residue-1, self.start_residue + len(self.sequence))

    def get_observable_residue_numbers(self):
        return self.observable_residue_numbers

    def calculate_observable_residue_numbers(self):
        # Only returns those residue numbers that are observable
        # (no N-terminal and no prolines)
        orns = self.get_residue_numbers()
        #remove N-terminal amides
        orns = orns[self.dataset.nfastamides:]

        orns2 = []

        for prn in range(self.dataset.nfastamides, len(self.sequence)):
            if self.sequence[prn] != 'P':
                orns2.append(prn+self.start_residue)

        #print("X", orns, orns2, self.get_residue_numbers())

        if len(orns2) != self.num_observable_amides:
            raise Exception("Peptide.get_observable_residue_numbers: Something is wrong with this calculation") 
        return orns2

    def get_id(self):
        return self.id

    def calculate_number_of_observable_amides(self):
        #number of observable amides is equal to peptide length - 2, minus remaining prolines
        num_prolines = self.sequence.count('P', self.dataset.nfastamides) + self.sequence.count('p', self.dataset.nfastamides)
        #print(self.sequence, len(self.sequence), num_prolines, self.dataset.nfastamides)
        return len(self.sequence) - num_prolines - self.dataset.nfastamides

    def get_number_of_observable_amides(self):
        return self.num_observable_amides

    def add_timepoint(self, time, sigma=None):
        if sigma is None:
            sigma = self.sigma
        if time not in [tp.time for tp in self.timepoints]:
            tp = Timepoint(time, sigma, peptide=self)
            self.timepoints.append(tp)
            return tp
        else:
            return self.get_timepoint_by_time(time)

    def get_timepoint_by_time(self, time):
        # Returns the timepoint where timepoint.time matches exactly.  Otherwise returns None
        for tp in self.timepoints:
            if tp.time == time:
                return tp
        return None

    def add_timepoints(self, times):
        for t in times:
            self.add_timepoint(t)

    def get_timepoints(self):
        return self.timepoints

    def get_chi_value(self, sig):
        '''
        Return chi score of model compared to experimental data
        '''
        chi=0
        totd=0
        nrep=0
        for t in self.timepoints:
            t.get_model_avg()
            t.get_model_sd()
            for r in t.replicates:
                if t.model_avg is not None:
                    if sig=="model":
                        sig=t.sigma
                    chi+=(r.deut-t.model_avg)**2/sig
                    nrep+=1
                #print f.seq, chi, t.time, t.model_avg, t.deut
        if len(self.timepoints) > 0:
            self.chi = chi/nrep
        else:
            self.chi = 0
        return self.chi

    def calculate_peptide_score_freq_grid(self, freq_grid, exp_grid, sig, save=False, force=True):
        score=0
        for tp in self.timepoints:
            tp.calc_model_deut(freq_grid, exp_grid, self.num_observable_amides)
            score += tp.calculate_tp_score(grid, exp_grid, sig, self.num_observable_amides, force_calc=True)
        return score        

    def calculate_score(self, grid, exp_grid, sig, save=False, force=False):
        score = 0     
        for tp in self.timepoints:
            tp.calc_model_deut(grid, exp_grid, self.num_observable_amides)
            score += tp.calculate_tp_score(grid, exp_grid, sig, self.num_observable_amides, force_calc=True)
        return score

    def set_timepoint_sigmas(self, sigma=None):
        if sigma == None:
            sigma = self.sigma
        for tp in self.timepoints:
            tp.sigma = sigma 

    def calculate_average_deuteration(self):
        # Calculates the average deuteration of the peptide over all timepoints
        vals = []
        for tp in self.timepoints:
            avg, sd = tp.get_avg_sd()
            vals.append(avg)

        return(numpy.average(vals))

    def set_sigma(self, sigma):
        self.sigma = sigma
        for tp in self.get_timepoints():
            tp.set_sigma(sigma)

    def set_back_exchange(self, value):
        self.back_exchange = value

    def set_sidechain_exchange(self, value):
        self.sidechain_exchange = value

class Timepoint(object):
    '''
    Timepoint objects are bound to a specific Fragment, where Timepoint represents
    a regular observation time of the HDX experiment.
    The class contains both the experimental data (as a list of Replicate objects)
    as well as a list of calculated values from the forward model (self.models)
    '''
    def __init__(self, time, sigma0, peptide=None):
        '''
        @param time - Time in seconds
        @param sigma0 - Initial estimate of timepoint error sigma in pctD units. 
        '''
        self.time = time
        self.models = []
        self.replicates = []
        self.sigma = sigma0
        self.peptide = peptide

    def get_score(self):
        score = 0
        for rep in self.replicates():
            score += rep.get_score()
        return score

    def number_of_replicates(self):
        return len(self.replicates)

    def add_replicate(self, deut, experiment_id=None, score=1.0, rt=None, charge_state=None, max_d=None, unique_id=None):
        if max_d is not None:
            self.replicates.append(Replicate(deut, experiment_id=experiment_id, reliability=score, rt=rt, charge_state=charge_state,
                                         timepoint=self, peptide=self.peptide, max_d=max_d, unique_id=unique_id))
        else:
            self.replicates.append(Replicate(deut, experiment_id=experiment_id, reliability=score, rt=rt, charge_state=charge_state,
                                         timepoint=self, peptide=self.peptide, unique_id=unique_id))

    def get_avg_sd(self):
        if len(self.replicates) < 1:
            #print "No replicates in timepoint, ", self.time
            self.avg=None
            self.sd=None
        elif len(self.replicates) <= 2:
            #print "Not enough replicates, ", self.time
            self.avg=sum(r.deut for r in self.replicates)/len(self.replicates)
            self.sd=None
        else:
            sum_deut=0
            sumsq_deut=0
            for r in self.replicates:
                sum_deut=sum_deut+float(r.deut)
                sumsq_deut=sumsq_deut+r.deut**2
            self.avg=sum_deut/len(self.replicates)
            self.sd=numpy.sqrt((len(self.replicates)*sumsq_deut-sum_deut**2)/(len(self.replicates)**2))
        return (self.avg, self.sd)

    def get_model_avg(self):
        if len(self.models) < 1:
            print("No models imported into timepoint ", self.time)
            self.model_avg=None
            return None
        else:
            #print "Not enough replicates, ", self.time
            self.model_avg=numpy.average(self.models)
        return self.model_avg

    def get_model_sd(self):
        if len(self.models) <= 2:
            print("Not enough models to calculate SD ", self.time)
            self.model_sd=None
        else:
            #print "Not enough replicates, ", self.time
            self.model_sd=numpy.std(self.models)
        return self.model_sd

    def add_model_deuteration_value(self, deut):
        self.models.append(deut)

    def set_deuteration(self, deut):
        self.model_deuteration = deut

    def get_model_deuteration(self):
        return self.model_deuteration

    def clear_model_deuteration_values(self):
        self.models=[]

    def remove_outliers(self, sigma, numsig=3):
        mean,sd=self.get_avg_sd()
        outliers=0
        for d in self.deut:
            if abs(d-mean)/sigma>numsig:
                self.deut.remove(d)
                print("outlier removed: ", d, mean, sigma)
                outliers=outliers+1
                self.num_replicates=self.num_replicates-1
        return outliers

    def get_sigma(self):
        return self.sigma

    def set_sigma(self, sigma):
        self.sigma = sigma

    def get_replicates(self):
        return self.replicates

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score
    
    # @property
    # def avg_iso_envelope(self):
    #     return tools.get_weighted_avg_iso_envelope(self)

    def calculate_avg_iso_envelope(self):
        #self.avg_iso_envelope = tools.get_weighted_avg_iso_envelope(self)
        sigma = tools.get_iso_sigma(self)
        self.set_sigma(sigma)
    

class SimulatedData(object):
    # Object that defines a simulated system and creates a simulated dataset from that system
    def __init__(self, sequence, peptides=None, truth=None, error=None):
        self.peptides = peptides # list of tuples,
        self.sequence = sequence 
        self.truth = truth

    def create_dataset(self):
        data = Dataset(sequence=self.sequence)
        #for p in self.peptides:
        #    print(p)


class Replicate(object):
    """ Replicate objects store the individual data observations
        @param deut - deuteration level in percentD
        @param rid - the deuterium concentration
        @param recovery - the 2D recovery estimate
    """
    def __init__(self, deut, experiment_id, reliability=1.0, rt=None, charge_state=None, timepoint=None, peptide=None, max_d=None, unique_id=None):
        self.deut = deut
        self.experiment_id = experiment_id
        self.reliability = reliability
        self.rt = rt
        self.charge_state = charge_state
        self.timepoint = timepoint 
        self.peptide = peptide
        self.unique_id = unique_id
        if max_d is None:
            self.max_d = max_d
        else:
            self.max_d = self.peptide.max_d

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_deut(self):
        return self.deut

    def set_prior_probability(self, prior):
        self.prior_probability = prior

    def load_raw_ms_csv(self, csv_file):
        df = pd.read_csv(csv_file, names=['m/z', 'Intensity'])
        # normalize intensity to sum to 1
        #df['Intensity'] = df['Intensity'] / df['Intensity'].sum()
        self.raw_ms = df
        iso = tools.get_isotope_envelope(self)
        if iso is not None:
            self.isotope_envelope = iso['Intensity'].values
        else:
            self.isotope_envelope = None

    # def raw_ms_to_isotope_envelope(self):
    #     self.isotope_envelope = tools.refine_raw_iso(self)




def merge_mutiple_datasets(datasets_list):

    names = [dataset.name for dataset in datasets_list]
    if len(set(names)) != 1:
        raise ValueError("Datasets must have the same name")
    
    if len(set([(dataset.conditions.temperature,dataset.conditions.pH,dataset.conditions.saturation) for dataset in datasets_list])) != 1:
        raise ValueError("Datasets must have the same condition")
    merged_datasets = Dataset(name=datasets_list[0].name, sequence=datasets_list[0].sequence, conditions=datasets_list[0].conditions)

    for dataset in datasets_list:
        for pep in dataset.peptides:
            
            pep.set_dataset(merged_datasets)
            new_peptide = merged_datasets.create_peptide(pep.sequence, pep.start_residue,peptide_id=pep.id)
            new_peptide.max_d = pep.max_d

            if new_peptide is not None:

                for tp in pep.get_timepoints():

                    # If timepoint is already in fragment, grab that timepoint
                    if tp.time in [tp.time for tp in new_peptide.get_timepoints()]:
                        tp_obj = new_peptide.get_timepoint_by_time(tp.time)
                    #If not, add a new timepoint at this time.  What to put for sigma??
                    else:
                        tp_obj = new_peptide.add_timepoint(tp.time)

                    for rep in tp.get_replicates():
                        tp_obj.replicates.append(rep)
                        

    merged_datasets.calculate_observable_rate_bounds()
    merged_datasets.calculate_observable_protection_factors()
    
    return merged_datasets


def generate_range_chunks(total_length, chunk_length=100, extra_length=20):

    chunks = []

    for i in range(math.ceil(total_length/chunk_length)):
        chunk_start = max(i*chunk_length-extra_length, 0)
        chunk_end = min((i+1)*chunk_length+extra_length, total_length)

        chunks.append((chunk_start, chunk_end))

    return chunks


def spilt_dataset_into_chunks(dataset, chunk_size=100):

    protein_size = len(dataset.sequence)
    range_chunks = generate_range_chunks(protein_size, chunk_length=chunk_size)

    chunked_datasets = []

    for i in range(len(range_chunks)):
        start, end = range_chunks[i]
        chunked_dataset = Dataset(name=dataset.name, sequence=dataset.sequence, conditions=dataset.conditions)
        
        for pep in dataset.peptides:
            if (start <= pep.start_residue <= end) or (start <= pep.end_residue <= end):
                pep.set_dataset(chunked_dataset)
                new_peptide = chunked_dataset.create_peptide(pep.sequence, pep.start_residue, peptide_id=pep.id)
                new_peptide.max_d = pep.max_d

                if new_peptide is not None:

                    for tp in pep.get_timepoints():

                        # If timepoint is already in fragment, grab that timepoint
                        if tp.time in [tp.time for tp in new_peptide.get_timepoints()]:
                            tp_obj = new_peptide.get_timepoint_by_time(tp.time)
                        #If not, add a new timepoint at this time.  What to put for sigma??
                        else:
                            tp_obj = new_peptide.add_timepoint(tp.time)

                        for rep in tp.get_replicates():
                            tp_obj.replicates.append(rep)
                            

        chunked_dataset.observable_bounds = dataset.observable_bounds
        chunked_dataset.observable_protection_factors = dataset.observable_protection_factors
    
        chunked_datasets.append(chunked_dataset)

    return range_chunks, chunked_datasets