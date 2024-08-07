"""
   Sampling functions for HDX models
"""
from __future__ import print_function
from bayesian_hdx_v2 import scoring
import numpy
import numpy.random
import math
import scipy
from bayesian_hdx_v2 import system
from random import shuffle
from numpy import linalg
from copy import deepcopy
import os
from itertools import combinations_with_replacement
from itertools import permutations
from bayesian_hdx_v2 import tools
from tqdm import tqdm
import MDAnalysis
import numpy as np
import random

# if exp(-diff/temp) > rand: accept

def metropolis_criteria(old_score, new_score, temp, proposal_ratio=0.4):
    # Returns True if move is accepted, False if move is not accepted
    if new_score <= old_score:
        return True
    rand_num = numpy.random.rand()
    boltzmann_prob = numpy.exp(-1*(new_score - old_score)/temp)
    #print("  MCMC", boltzmann_prob * proposal_ratio > rand_num, boltzmann_prob, rand_num, old_score-new_score, temp)
    if boltzmann_prob * proposal_ratio > rand_num:
        # Upward step taken
        return True
    else:
        return False

# Connect this object to a Residue
class SampledInt(object):
    def __init__(self, allowed_range, random=True, adjacency=3, is_sampled=True):
        self.range=allowed_range # range of values this integer can be
        self.random=random # Are we doing random assignment?
        self.adjacency=adjacency # What is the range of values that delta can be?
        self.moved=False
        #self.index=initialize()

    def initialize():
        self.propose_move()

    def set_adjacency(self, is_adjacency, adjacency):
        if is_adjacency:
            self.random = False
            self.adjacency = adjacency
        else:
            self.random = True

    def set_range(self, allowed_range):
        self.range=allowed_range

    def set_moved(self):
        self.moved=True

    def propose_move(self, previous):
        if self.moved:
            print("This mover has already been moved")

        if self.random:
            # Remove the previous value from the list of potential moves
            this_range = [s for s in self.range if s != previous]
            new_index = numpy.random.randint(0, len(this_range))

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Instead of this, change the value in the residue object
            # self.object.set_value(new_index)
            return this_range[new_index]

        else:
            self.old_index = self.range.index(previous)
            new_index = -1
            while new_index < 0 or new_index >= len(self.range):
                sign = numpy.random.randint(0,2) * 2 - 1
                magnitude = numpy.random.randint(1, self.adjacency+1)
                new_index = int(self.old_index + magnitude * sign)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Instead of this, change the value in the residue object
            # self.object.set_value(new_index)
            return self.range[new_index]

        self.moved=True

    def accept(self):
        # Keep the residue object the same
        self.moved=False

    def reject(self):
        # Change the residue object back to its original value
        # NOT IMPLEMENTED!!!
        # self.object.set_value(self.range[self.old_index])
        self.moved=False

class SampledIntCircular(SampledInt):
    def propose_move(self, previous, lower_bound=None, upper_bound=None):
        if self.moved:
            print("This mover has already been moved")

        if self.random:
            # Remove the previous value from the list of potential moves
            this_range = [s for s in self.range if s != previous]
            new_index = numpy.random.randint(0, len(this_range))

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Instead of this, change the value in the residue object
            # self.object.set_value(new_index)
            return this_range[new_index]

        else:
            self.old_index = self.range.index(previous)
            sign = numpy.random.randint(0, 2) * 2 - 1
            magnitude = numpy.random.randint(1, self.adjacency + 1)
            if lower_bound is None and upper_bound is None:
                new_index = (self.old_index + magnitude * sign) % len(self.range)
            else:
                new_index = (self.old_index + magnitude * sign) % (upper_bound - lower_bound) + lower_bound
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Instead of this, change the value in the residue object
            # self.object.set_value(new_index)
            return self.range[new_index]

        self.moved=True

class SampledFloat(object):
    def __init__(self, lower_bound, upper_bound, maxdel, distribution="uniform", random=False, is_sampled=True):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.maxdel = maxdel
        self.random = random

    def propose_move(self, previous):
        if self.random:
            new_value = numpy.random.rand() * (self.upper_bound - self.lower_bound) + self.lower_bound
            return new_value

        else:
            new_value = self.lower_bound - 1
            #print(self.upper_bound, self.lower_bound)
            while new_value >= self.upper_bound or new_value <= self.lower_bound:
                sign = numpy.random.randint(0,2) * 2 - 1
                magnitude = numpy.random.rand() * self.maxdel 
                new_value = previous + sign * magnitude
            #print("PROPOSE_MOVE:", new_value, previous, sign, magnitude, self.upper_bound, self.lower_bound)
            return new_value
            '''
            if new_value <= self.upper_bound and new_value >= self.lower_bound:
                print("PROPOSE_MOVE:", new_value, previous, sign, magnitude, self.upper_bound, self.lower_bound)
                return new_value
            else:
                print("NO_MOVE")
                self.propose_move(previous)
            '''

class EnumerationSampler(object):
    def __init__(self, sys, sigma_sample_level=None):
        if type(sys) is system.State:
            states = [sys]
        elif type(sys) is system.Macromolecule:
            states = sys.get_states()
        elif type(sys) is system.System:
            states = sum([mol.get_states() for mol in sys.get_macromolecules()], [])
        else:
            raise Exception("You must pass a System, State or Macromolecule object")

        self.states = states 
        
        m = self.states[0].output_model 

    def uun(self, write=False):
        """Enumerates and scores all possible models for the given an exp_grid
        returns the top num_models scoring exp grids"""
        output = self.states[0].macromolecule.system.get_output()
        acceptance = 0.0
        # Get the residue types
        for s in self.states:
            s.initialize(1)

            if write:
                output_file = open(output.get_output_file(s), "a")

            resis = s.observed_residues

            n = len(resis) 
            exp_grid = s.output_model.get_sampling_grid()
            nbin = len(exp_grid)
            num = n
            possible_number_combinations = list(combinations_with_replacement(range(1,nbin+1), n))
            #all_possible_combinations = list(product(range(n), repeat=nbin))
            print("Assessing", possible_number_combinations, "combinations")
            for model in possible_number_combinations:

                for n in range(len(resis)):
                    resid = resis[n]
                    mod_val = model[n]
                    s.output_model.change_residue(resid, mod_val)
                    s.change_single_residue_incorporation(resid, int(mod_val))

                score = s.calculate_score(s.output_model.model)

                print(model, score)
                if write:
                    output.write_model_to_file(output_file, s, s.output_model.get_masked_model(s.get_observed_residues()), score, acceptance, sigmas=True)
            if write:
                output_file.close()

class MCSampler(object):
    '''
    # 
    
    '''
    def __init__(self, sys, initialize=True, 
                if_sample_centroid_sigma=False,
                if_sample_envelope_sigma=False,
                if_sample_back_exchange=False,
                if_sample_sidechain_exchange=False,
                pct_moves=25, 
                accept_range=(0.3, 0.8)):
        # Ensure that all states in system has a dataset and a model and a scoring function
        '''
        @param sigma_sample_level - None: Don't sample sigmas. "dataset": sample one sigma per dataset. 
                            "peptide": 1 sigma per peptide, "timepoint": 1 sigma per timepoint
                            "replicate": 1 sigma per replicate_id
        '''
        if type(sys) is system.State:
            states = [sys]
        elif type(sys) is system.Macromolecule:
            states = sys.get_states()
        elif type(sys) is system.System:
            states = sum([mol.get_states() for mol in sys.get_macromolecules()], [])
        else:
            raise Exception("You must pass a System, State or Macromolecule object")

        self.states = states

        m = self.states[0].output_model

        if m.sampler_type == "int":
            self.residue_sampler = SampledInt(m.sampler_range())
            #self.residue_sampler = SampledIntCircular(m.sampler_range())
        elif m.sampler_type == "float":
            raise Exception("Sampler.__init__: Floating point representation for residues is not implemented yet")
        
        self.if_sample_centroid_sigma = if_sample_centroid_sigma
        self.if_sample_envelope_sigma = if_sample_envelope_sigma
        self.if_sample_back_exchange = if_sample_back_exchange
        self.if_sample_sidechain_exchange = if_sample_sidechain_exchange
        self.centroid_sigma_sampler = SampledFloat(0.1, 2, 0.05)
        self.envelope_sigma_sampler = SampledFloat(0.1, 1, 0.05)
        self.back_exchange_sampler = SampledFloat(0.0, 0.5, 0.05)
        self.sidechain_exchange_sampler = SampledFloat(0, 0.3, 0.05)
        self.pct_moves = pct_moves
        self.acceptance_range = accept_range
        # Recalculates all sectors and timepoint data.
        if initialize:
            for s in self.states:
                s.initialize()

        self.structural_prior = None

    def set_structural_prior(self, input_pdb):
        
        def find_peptide(seq, peptide):
            start_index = seq.find(peptide)
            if start_index == -1:
                return (-1, -1)
            end_index = start_index + len(peptide) - 1
            return (start_index, end_index)
        
        def pdb2seq(pdb_file):

            import warnings
            from Bio import SeqIO
            from Bio import BiopythonWarning

            # Suppress all Biopython-specific warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', BiopythonWarning)
                records = list(SeqIO.parse(pdb_file, 'pdb-atom'))
                return str(records[0].seq)

        
        pdb_sequence = pdb2seq(input_pdb)
        a_middle_pep = self.states[0].data[0].get_sequence()[80:90]
        
        pdb_start, pdb_end= find_peptide(pdb_sequence, a_middle_pep)
        index_offset = 80 - pdb_start 


        def get_if_exposed(pdb_file,):
            
            import warnings
            from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
            
            warnings.filterwarnings("ignore")
            
            u = MDAnalysis.Universe(pdb_file,)
            
            protein = u.select_atoms('protein')

            if_exposed = []
            for res in protein.residues:

                hbonds = HBA(universe=u, d_a_cutoff=3.5, d_h_a_angle_cutoff=30)
                hbonds.donors_sel = f'protein and name N and resid {res.resid}'
                # caution, in pdb file generate by mdtraj, HNs are renamed to H
                hbonds.hydrogens_sel = 'protein and name H'
                hbonds.acceptors_sel = 'protein and name O'
                hbonds.run()
                
                if hbonds.results.hbonds.size == 0:
                    if_exposed.append(1)
                else:
                    if_exposed.append(0)
                
            return np.array(if_exposed)


        if_exposed_array = get_if_exposed(input_pdb)
        
        sample_range_list = []
        for i in range(len(self.states[0].data[0].get_sequence())):
            res_i_in_pdb = i - index_offset
            if res_i_in_pdb >= 0 and if_exposed_array[res_i_in_pdb] == 1:
                upper_bound = np.where(np.linspace( 0, 14, self.states[0].output_model.grid_size )<3)[0][-1]
                sample_range_list.append((0, upper_bound))
            else:
                sample_range_list.append((0, self.states[0].output_model.grid_size ))
        
        self.structural_prior = sample_range_list


    def run_exponential_temperature_decay(self, tmax=100, tmin=2.0, 
                                annealing_steps=200, 
                                steps_per_anneal=1, 
                                write=False, 
                                adjacency=10):
        '''
        A simulated annealing that starts from high temperature and gradually cools
        to a low temperature
        '''
        import math

        # Set how far each residue can move.
        self.residue_sampler.set_adjacency(True, adjacency)

        print("********")
        print("Starting Exponential Temperature Decay from")
        print("T =", tmax, "to T =", tmin, "over", annealing_steps, "steps.")
        print("********")
        deltaT = math.exp(math.log(tmin/tmax)/annealing_steps)
        for s in range(annealing_steps):
            Tm = tmax * (deltaT ** s)
            for i in range(steps_per_anneal):
                score, model_avg, accept = self.run_one_step(Tm, write)

            print('Temp: %2.2f | Score: %2.1f' % (Tm,score))


    def run_benchmark(self):
        import time
        print("Starting Benchmark")
        '''Run 100 MC steps and report the time to run 1000 steps
        '''       
        times = []
        for i in range(10):
            start = time.time()
            self.run(NSTEPS=10)
            end = time.time()
            times.append((end-start)*100)
        time = numpy.average(times)
        sd = numpy.std(times)
        print("This system will take about ", int(time), "+/-", int(sd*2) , " seconds per 1000 steps")

    def get_acceptable_temperature(self, init_temp, acceptance_range, nsteps=20, alpha=0.85):
        accept = False
        direction = 0  # 0: init, 1: increased, -1: decreased
        
        orig_pct_moves = self.pct_moves
        self.pct_moves = 100 

        temp = init_temp

        while not accept:
            acceptance_total = 0.0

            for i in range(nsteps):
                score, model_avg, acceptance = self.run_one_step_swap(temp)
                acceptance_total += acceptance

            acceptance_ratio = acceptance_total / nsteps

            if acceptance_ratio < acceptance_range[0]:
                # Acceptance ratio too low, need to increase temperature
                if direction == -1:  # If the last change was a decrease, accelerate the change
                    alpha *= 0.5
                temp *= (1 + alpha)
                direction = 1
            elif acceptance_ratio > acceptance_range[1]:
                # Acceptance ratio too high, need to decrease temperature
                if direction == 1:  # If the last change was an increase, accelerate the change
                    alpha *= 0.5
                temp *= (1 - alpha)
                direction = -1
            else:
                accept = True

            print(accept, temp, acceptance_ratio)

        self.pct_moves = orig_pct_moves 
        self.temperature = temp
        return temp

    def run(self, NSTEPS=100, init_temp=10, write=False, write_all=False, acceptance_range=None, find_temperature=False, 
            if_using_swap=True):
        print("Starting run")
        if acceptance_range is None:
            acceptance_range = self.acceptance_range
        acceptance_total = 1.0       
        init_score = 0
        output_files = []
        output = self.states[0].macromolecule.system.get_output()
        for s in self.states:
            state_score = s.calculate_score(s.output_model.model)
            s.set_score(state_score)
            init_score += state_score
            #print(s, state_score)
            if write:
                output_files.append(open(output.get_output_file(s), "a"))
            for d in s.data:
                d.collect_times()
        if find_temperature:
            print("Finding initial running temperature")
            temperature = self.get_acceptable_temperature(init_temp=init_temp, acceptance_range=acceptance_range)
        else:
            temperature = init_temp

        print("Simulation temperature =", temperature)
        print("Step score | states_avg_protection_factor | mc_acceptance_ratio")
        for i in tqdm(range(NSTEPS)):
            #print("Step:", i)
            if if_using_swap:
                score, model_avg_str, acceptance = self.run_one_step_swap(temperature, write_all)
            else:
                score, model_avg_str, acceptance = self.run_one_step(temperature, write_all)
            acceptance_total += acceptance
            if i%50 == 0:
                tqdm.write(f"Step {i}, {score:.1f} | {model_avg_str}, {acceptance:.2f}")

            if write:
                for s in range(len(self.states)):
                    st = self.states[s]
                    if st.output_model.sample_only_observed_residues == True:
                        model = st.output_model.get_masked_model(st.get_observed_residues())
                    else:
                        model = st.output_model.get_model()

                    output.write_model_to_file(output_files[s], st, model, st.score, acceptance, sigmas=False, back_exchange=True, sidechain_exchange=True)

        acceptance_ratio = acceptance_total/NSTEPS
        print("Average acceptance ratio for this run = ", acceptance_ratio, " |  Temp = ", temperature)

        for of in output_files:
            of.close()  

    def run_one_step(self, temperature, write=False):
        # Running one MC step over all model states and sigma parameters
        # Loop over all states in self.states
        total_score = 0
        acceptance_ratio = 0

        for state in self.states:

            init_model = deepcopy(state.output_model.model)
            init_score = state.get_score()

            ###########################
            # This should be movable particles
            ###########################

            if state.output_model.sample_only_observed_residues==True:
                resis = state.observed_residues
            else:
                resis = state.get_exchanging_residues()

            shuffle(resis)
            flips = int(max(math.ceil((self.pct_moves * len(resis))/100.), 1))
            # Flip a number of residues

            for r in resis[:flips]:
                # Get the sector that holds this residue           
                #r_sector = state.residue_sector_dictionary[r]
                # Get the current value for this residue
                oldval = int(state.output_model.get_model_residue(r))
                # Propose a new value given the current state
                oldscore = state.get_score()
                #print(r, oldval, oldscore, state.output_model.get_model())
                if self.structural_prior is not None:
                    newval = self.residue_sampler.propose_move(oldval, lower_bound=self.structural_prior[r][0], upper_bound=self.structural_prior[r][1])
                else:
                    newval = self.residue_sampler.propose_move(oldval) 

                # Change the residue incorporation values in each sector and calculate the new score:
                state.change_single_residue_incorporation(r, newval)
                newscore = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
                state.set_score(newscore)

                accept = metropolis_criteria(oldscore, newscore, temperature)

                if not accept:
                    flips -= 1
                    state.output_model.change_residue(r, oldval)
                    state.change_single_residue_incorporation(r, oldval)
                    state.set_score(oldscore)

            # Determine the total number of moves performed
            flips2=0
            for i in range(len(init_model)):
                if init_model[i] != state.output_model.model[i]:
                    flips2 += 1

                # Measures the total change in Pf value over all residues. Might be useful.
                #m_sq_change = sum((state.output_model.model - init_model)**2)


            # Now sample the sigma values if we are doing that
            if self.sigma_sample_level is not None:
                for d in state.data:
                    self.sample_sigma(state, temperature)

            # Recalculate the state score and add it to the total score
            final_state_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
            state.set_score(final_state_score)
            total_score += final_state_score

            '''
            mpf = state.output_model.get_current_model()
            tot_mpf = 0
            num_mpf = 0
            for pep in state.data[0].get_peptides():
                for i in pep.get_observable_residue_numbers():
                    if not math.isnan(mpf[i-1]) and mpf[i-1] != numpy.inf:
                        tot_mpf += mpf[i-1]
                        num_mpf += 1 
            '''

            acceptance_ratio += float(flips2)/int(max(math.ceil((self.pct_moves * len(resis))/100.), 1))

        model_avg = [numpy.average(state.output_model.get_current_model()) for s in self.states]
        model_avg_str = ""
        for m in model_avg:
            model_avg_str+=str(m)+" "

        return total_score, model_avg_str[0:-1], acceptance_ratio/len(self.states)#, m_sq_change #acceptance_ratio/len(self.states)

    def run_one_step_swap(self, temperature, write=False):
        # Running one MC step over all model states and sigma parameters
        # Loop over all states in self.states
        total_score = 0
        acceptance_ratio = 0

        for state in self.states:

            init_model = deepcopy(state.output_model.model)

            ###########################
            # This should be movable particles
            ###########################

            if state.output_model.sample_only_observed_residues==True:
                resis = state.observed_residues
            else:
                resis = state.get_exchanging_residues()

            shuffle(resis)
            flips = int(max(math.ceil((self.pct_moves * len(resis))/100.), 1))
            # Flip a number of residues

            for i in range(flips):
            #for i in range(len(resis)):
                if i < len(resis) - 1:
                    # Select two adjacent residues for swap
                    r1 = resis[i]
                    r2 = resis[i + 1]

                    # Get the current values for these residues
                    oldval1 = int(state.output_model.get_model_residue(r1))
                    oldval2 = int(state.output_model.get_model_residue(r2))
                    oldscore = state.get_score()
                    old_rep_score = state.all_rep_data['rep_score'].copy()

                    # Swap the values
                    state.output_model.change_residue(r1, oldval2)
                    state.output_model.change_residue(r2, oldval1)
                    state.change_single_residue_incorporation(r1, oldval2)
                    state.change_single_residue_incorporation(r2, oldval1)

                    # Calculate the new score after swapping
                    newscore = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
                    state.set_score(newscore)

                    # Decide whether to accept the swap
                    # greedy metropolis criteria
                    accept = metropolis_criteria(oldscore, newscore, 0.0)

                    if not accept:
                        # Revert the swap if not accepted
                        state.output_model.change_residue(r1, oldval1)
                        state.output_model.change_residue(r2, oldval2)
                        state.change_single_residue_incorporation(r1, oldval1)
                        state.change_single_residue_incorporation(r2, oldval2)
                        state.set_score(oldscore)
                        state.all_rep_data['rep_score'] = old_rep_score


            for r in resis[:flips]:
                # Get the sector that holds this residue           
                #r_sector = state.residue_sector_dictionary[r]
                # Get the current value for this residue
                oldval = int(state.output_model.get_model_residue(r))
                # Propose a new value given the current state
                oldscore = state.get_score()
                old_rep_score = state.all_rep_data['rep_score'].copy()
                #print(r, oldval, oldscore, state.output_model.get_model())
                newval = self.residue_sampler.propose_move(oldval) 

                # Change the residue incorporation values in each sector and calculate the new score:
                state.change_single_residue_incorporation(r, newval)
                newscore = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
                state.set_score(newscore)

                accept = metropolis_criteria(oldscore, newscore, temperature)

                if not accept:
                    flips -= 1
                    state.output_model.change_residue(r, oldval)
                    state.change_single_residue_incorporation(r, oldval)
                    state.set_score(oldscore)
                    state.all_rep_data['rep_score'] = old_rep_score

            # Determine the total number of moves performed
            flips2=0
            for i in range(len(init_model)):
                if init_model[i] != state.output_model.model[i]:
                    flips2 += 1

                # Measures the total change in Pf value over all residues. Might be useful.
                #m_sq_change = sum((state.output_model.model - init_model)**2)


            # Now sample the sigma values if we are doing that
            if self.if_sample_centroid_sigma or self.if_sample_envelope_sigma:
                # for d in state.data:
                if random.random() < 0.2:
                    self.sample_sigma(state, temperature/10)

            if self.if_sample_back_exchange:
                if random.random() < 0.2:
                    self.sample_back_exchange(state, temperature/10)

            if self.if_sample_sidechain_exchange:
                #if random.random() < 0.2:
                self.sample_sidechain_exchange(state, 0)

            # Recalculate the state score and add it to the total score
            final_state_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
            state.set_score(final_state_score)
            total_score += final_state_score

            '''
            mpf = state.output_model.get_current_model()
            tot_mpf = 0
            num_mpf = 0
            for pep in state.data[0].get_peptides():
                for i in pep.get_observable_residue_numbers():
                    if not math.isnan(mpf[i-1]) and mpf[i-1] != numpy.inf:
                        tot_mpf += mpf[i-1]
                        num_mpf += 1 
            '''

            acceptance_ratio += float(flips2)/int(max(math.ceil((self.pct_moves * len(resis))/100.), 1))
            #acceptance_ratio += float(flips2)/int(max(math.ceil((self.pct_moves * len(resis)+len(resis))/100.), 1))

        model_avg = [numpy.average(state.output_model.get_current_model()) for s in self.states]
        model_avg_str = ""
        for m in model_avg:
            model_avg_str+=str(m)+" "

        return total_score, model_avg_str[0:-1], acceptance_ratio/len(self.states)#, m_sq_change #acceptance_ratio/len(self.states)



    def sample_sigma(self, state, temperature):
        # Sample the sigma values in this dataset.
        # Returns the acceptance boolean

        # centroid sigma
        if self.if_sample_centroid_sigma:
            init_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
            init_sigma = state.scoring_function.forward_model.centroid_sigma
            new_sigma = self.centroid_sigma_sampler.propose_move(init_sigma)

            state.scoring_function.forward_model.centroid_sigma = new_sigma
            new_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())

            if not metropolis_criteria(init_score, new_score, temperature):
                # Reset the sigma back to the original one
                state.scoring_function.forward_model.centroid_sigma = init_sigma
                state.set_score(init_score)

        # envelope sigma
        if self.if_sample_envelope_sigma:
            init_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
            init_sigma = state.scoring_function.forward_model.envelope_sigma
            new_sigma = self.envelope_sigma_sampler.propose_move(init_sigma)

            state.scoring_function.forward_model.envelope_sigma = new_sigma
            new_score = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())

            if not metropolis_criteria(init_score, new_score, temperature):
                # Reset the sigma back to the original one
                state.scoring_function.forward_model.envelope_sigma = init_sigma
                state.set_score(init_score)


    def sample_back_exchange(self, state, temperature):
        # Sample the back exchange values in this dataset.
        # Returns the acceptance boolean

        #random_peptides = random.sample(state.get_all_peptides(),  int(len(state.get_all_peptides()) * self.pct_moves / 100))

        for pep in state.get_all_peptides():
            init_score = state.calculate_peptides_score([pep], state.output_model.get_current_model())
            init_rep_score = state.all_rep_data['rep_score'].copy()
            init_back_exchange = pep.back_exchange

            exp_back_exchange = 1 - pep.max_d/pep.num_observable_amides*state.data[0].conditions.saturation
            self.back_exchange_sampler.upper_bound = min(exp_back_exchange + 0.3, 1)
            self.back_exchange_sampler.lower_bound = max(exp_back_exchange - 0.3, 0)
            new_back_exchange = self.back_exchange_sampler.propose_move(init_back_exchange)
            # exp_back_exchange = 1 - pep.max_d/pep.num_observable_amides*state.data[0].conditions.saturation
            # new_back_exchange = random.uniform(exp_back_exchange-0.3, exp_back_exchange+0.3)

            pep.back_exchange = new_back_exchange
            new_score = state.calculate_peptides_score([pep], state.output_model.get_current_model())

            if not metropolis_criteria(init_score, new_score, temperature):
                # Reset the back exchange back to the original one
                pep.back_exchange = init_back_exchange
                state.set_score(init_score)
                state.all_rep_data['rep_score'] = init_rep_score


    def sample_back_exchange_residue_level(self, state, temperature):

        if state.output_model.sample_only_observed_residues:
            resis = state.observed_residues
        else:
            resis = state.get_exchanging_residues()


        for r in resis:
            
            oldval_pf = int(state.output_model.get_model_residue(r))
            oldval = int(state.output_model.get_model_back_exchange_residue(r))
            # Propose a new value given the current state
            oldscore = state.get_score()
            old_rep_score = state.all_rep_data['rep_score'].copy()
            #print(r, oldval, oldscore, state.output_model.get_model())
            newval = self.back_exchange_sampler.propose_move(oldval)

            # # Change the residue incorporation values in each sector and calculate the new score:
            state.output_model.model_back_exchange[r-1] = newval
            # newscore = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model())
            newscore = state.calculate_peptides_score(state.get_all_peptides(), state.output_model.get_current_model(), state.output_model.get_model_back_exchange())
            state.set_score(newscore)

            accept = metropolis_criteria(oldscore, newscore, temperature)

            if not accept:
                state.output_model.model_back_exchange[r-1] = oldval
                state.set_score(oldscore)
                state.all_rep_data['rep_score'] = old_rep_score


    def sample_sidechain_exchange(self, state, temperature, sample_level='dataset'):

        # peptide level
        if sample_level == 'peptide':
            for pep in state.get_all_peptides():
                init_score = state.calculate_peptides_score([pep], state.output_model.get_current_model())
                init_rep_score = state.all_rep_data['rep_score'].copy()
                init_sidechain_exchange = pep.sidechain_exchange
                new_sidechain_exchange = self.sidechain_exchange_sampler.propose_move(init_sidechain_exchange)

                pep.sidechain_exchange = new_sidechain_exchange
                new_score = state.calculate_peptides_score([pep], state.output_model.get_current_model())

                if not metropolis_criteria(init_score, new_score, temperature):
                    # Reset the sidechain exchange back to the original one
                    pep.sidechain_exchange = init_sidechain_exchange
                    state.set_score(init_score)
                    state.all_rep_data['rep_score'] = init_rep_score

        # dataset level
        if sample_level == 'dataset':
            peptides = state.get_all_peptides()
            init_score = state.calculate_peptides_score(peptides, state.output_model.get_current_model())
            init_rep_score = state.all_rep_data['rep_score'].copy()
            init_sidechain_exchange = peptides[0].sidechain_exchange

            new_sidechain_exchange = self.sidechain_exchange_sampler.propose_move(init_sidechain_exchange)

            for pep in peptides:
                pep.sidechain_exchange = new_sidechain_exchange
            new_score = state.calculate_peptides_score(peptides, state.output_model.get_current_model())

            if not metropolis_criteria(init_score, new_score, temperature):
                # Reset the sidechain exchange back to the original one
                for pep in peptides:
                    pep.sidechain_exchange = init_sidechain_exchange
                state.set_score(init_score)
                state.all_rep_data['rep_score'] = init_rep_score



def benchmark(model, sample_sigma):
    import time
    '''Run 100 MC steps and report the time to run 1000 steps
    '''
    times=[]
    for i in range(10):
        start=time.time()
        do_mc_sampling(model, NSTEPS=10, print_t=1000)
        end=time.time()
        times.append((end-start)*100)
    time=numpy.average(times)
    sd=numpy.std(times)
    print("This system will take about ", int(time), "+/-", int(sd*2) , " seconds per 1000 steps")


def simulated_annealing(model, sigma, sample_sig=True, equil_steps=10000, annealing_steps=100, save_all=False, 
                        outdir="./sampling_output/", outfile_prefix="", print_t=10, sample_sigma=False, noclobber=False):

    for temp in [20, 10, 5, 1]:
        do_mc_sampling(model, temp, sigma, NSTEPS=annealing_steps, sample_sigma=sample_sig, print_t=print_t,
                        save_results=save_all, outdir=outdir, outfile_prefix=outfile_prefix, noclobber=noclobber)

    # Equilibrium run  
    do_mc_sampling(model, 1.0, sigma, NSTEPS=equil_steps, sample_sigma=sample_sigma, print_t=print_t,
                        save_results=True, outdir=outdir, outfile_prefix=outfile_prefix, noclobber=noclobber)


def enumerate_fragment(frag, exp_grid, sig, num_models = 1):
    """Enumerates and scores all possible models for the given an exp_grid
    returns the top num_models scoring exp grids"""
    n = frag.num_observable_amides  
    nbin = len(exp_grid)
    num = n
    possible_number_combinations = list(combinations_with_replacement(range(n), nbin))
    #all_possible_combinations = list(product(range(n), repeat=nbin))
    score = 0
    minscore = 10000
    for i in possible_number_combinations:
        if sum(i) == n:
            numbers = list(i)
            all_possible_permutations = permutations(numbers)
            seen = set()
            for grid in all_possible_permutations:
                if grid in seen:
                    continue
                seen.add(grid)
                score = frag.calculate_frag_score(grid, exp_grid, sig, force=True)
                #print(frag.seq, score, grid)
                sum_exp = 0
                for x in range(len(exp_grid)):
                    sum_exp += exp_grid[x]*grid[x]
                if score < minscore:
                    minmodel = grid
                    minscore = score

    #print(numpy.array(minmodel), minscore)
    return numpy.array(minmodel)

