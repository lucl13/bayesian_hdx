"""
   Useful tools for HDX and MS data analysis

   Can be used as a standalone library outside of IMP 
   and the RResolvedHX
"""

from __future__ import print_function
import numpy
import numpy.random
import scipy
import math

AA_list = "ARNDCQEGHILKMFPSTWYV"

def get_residue_neighbor_effects(AA, pDcorr, T):
    # For each residue, a tuple containing:
    # 0:Milne acid lambda
    # 1:Milne acid rho
    # 2:Milne base lambda
    # 3:Milne base rho

    R = 1.987

    # Calculate Temp dependent pKa's
    pK_D = -1 * numpy.log10( 10**(-1*4.48)*numpy.exp(-1*1000*((1.0/T-1.0/278)/R)) )
    pK_E = -1 * numpy.log10( 10**(-1*4.93)*numpy.exp(-1*1083*((1.0/T-1.0/278)/R)) )
    pK_H = -1 * numpy.log10( 10**(-1*7.42)*numpy.exp(-1*7500*((1.0/T-1.0/278)/R)) )

    eff_dict = {
    "A" : (0.0,0.0,0.0,0.0),
    "R" : (-0.59,-0.32,0.07671225,0.22),
    "N" : (-0.58,-0.13,0.49,0.32),
    "C" : (-0.54,-0.46,0.62,0.55),
    "Q" : (-0.47,-0.27,0.06,0.20),
    "G" : (-0.22,0.21817605,0.26725157,0.17),
    "I" : (-0.91,-0.59,-0.73,-0.23),
    "L" : (-0.57,-0.13,-0.57625273,-0.21),
    "K" : (-0.56,-0.29,-0.04,0.12),
    "M" : (-0.64,-0.28,-0.00895484,0.11),
    "F" : (-0.52,-0.43,-0.23585946,0.06313159),
    "P" : ("",-0.19477347,"",-0.24),
    "S" : (-0.43799228,-0.38851893,0.37,0.29955029),
    "T" : (-0.79,-0.46807313,-0.06625798,0.20),
    "W" : (-0.40,-0.44,-0.41,-0.11),
    "Y" : (-0.41,-0.37,-0.27,0.05),
    "V" : (-0.73902227,-0.30,-0.70193448,-0.14),
    "NT" : ("",-1.32,"",1.62)
    }

    # Ionizable AA data from
    # Susumu Mori, Peter C.M. van Zijl, and David Shortle
    # PROTEINS: Structure, Function, and Genetics 28:325-332 (1997)

    if AA=="D":
        ne0 = numpy.log10(10**(-0.9-pDcorr)/(10**(-pK_D)+10**(-pDcorr))+10**(0.9-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne1 = numpy.log10(10**(-0.12-pDcorr)/(10**(-pK_D)+10**(-pDcorr))+10**(0.58-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne2 = numpy.log10(10**(0.69-pDcorr)/(10**(-pK_D)+10**(-pDcorr))+10**(0.1-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
        ne3 = numpy.log10(10**(0.6-pDcorr)/(10**(-pK_D)+10**(-pDcorr))+10**(-0.18-pK_D)/(10**(-pK_D)+10**(-pDcorr)))
    elif AA=="E":
        ne0 = numpy.log10(10**(-0.6-pDcorr)/(10**(-pK_E)+10**(-pDcorr))+10**(-0.9-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne1 = numpy.log10(10**(-0.27-pDcorr)/(10**(-pK_E)+10**(-pDcorr))+10**(0.31-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne2 = numpy.log10(10**(0.24-pDcorr)/(10**(-pK_E)+10**(-pDcorr))+10**(-0.11-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne3 = numpy.log10(10**(0.39-pDcorr)/(10**(-pK_E)+10**(-pDcorr))+10**(-0.15-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
    elif AA=="H":
        ne0 = numpy.log10(10**(-0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))+10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne1 = numpy.log10(10**(-0.51-pDcorr)/(10**(-pK_H)+10**(-pDcorr))+10**(0-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne2 = numpy.log10(10**(0.8-pDcorr)/(10**(-pK_H)+10**(-pDcorr))+10**(-0.1-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
        ne3 = numpy.log10(10**(0.83-pDcorr)/(10**(-pK_H)+10**(-pDcorr))+10**(0.14-pK_H)/(10**(-pK_H)+10**(-pDcorr)))
    elif AA=="CT":
        ne0 = numpy.log10(10**(0.05-pDcorr)/(10**(-pK_E)+10**(-pDcorr))+10**(0.96-pK_E)/(10**(-pK_E)+10**(-pDcorr)))
        ne1 = ""
        ne2 = -1.8
        ne3 = ""
    else:
        (ne0, ne1, ne2, ne3) = eff_dict.get(AA)

    #print(AA, pDcorr, (ne0, ne1, ne2, ne3))

    return (ne0, ne1, ne2, ne3)


ResidueChemicalContent = {
    # Tuple containing number of atoms of
    #  (Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur)
    #  for a free amino aicd

    "A" : (3,5,1,1,0),
    "R" : (6,12,4,1,0),
    "N" : (4,6,2,2,0),
    "D" : (4,5,1,3,0),
    "C" : (3,5,1,1,1),
    "Q" : (5,8,2,2,0),
    "E" : (5,7,1,3.0),
    "G" : (2,3,1,1,0),
    "H" : (6,7,3,1,0),
    "I" : (6,11,1,1,0),
    "L" : (6,11,1,1,0),
    "K" : (6,12,2,1,0),
    "M" : (5,9,1,1,1),
    "F" : (9,9,1,1,0),
    "P" : (5,7,1,1,0),
    "S" : (3,5,1,2,0),
    "T" : (4,7,1,2,0),
    "W" : (11,10,2,1,0),
    "Y" : (9,9,1,2,0),
    "V" : (5,9,1,1,0),
    "CT" : (0,1,0,1,0),
    "NT" : (0,1,0,0,0)
    # Eventually add AA modifications
}

ElementMasses = {
    # List of tuples containing the mass and abundance of atoms
    # in biological peptides (C, H, N, O, S)
    #
    # Data from https://www.ncsu.edu/chemistry/msf/pdf/IsotopicMass_NaturalAbundance.pdf
    #
    # Original references: 
    # The isotopic mass data is from:
    #   G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 
    #   G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480. 
    # The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic
    #   Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607. 

    "C" : [(12.000000, 98.93), (13.003355, 1.07)],
    "H" : [(1.007825, 99.9885), (2.14101, 0.0115)],
    "N" : [(14.0030764, 99.632), (15.000109, 0.368)],
    "O" : [(15.994915, 99.757), (16.999132, 0.038), (17.999160, 0.205)],
    "S" : [(31.972071, 94.93), (32.97158, 0.76), (33.967867, 4.29), (35.967081, 0.02)]
}



def calc_intrinsic_rate(Laa, Raa, pH, T, La2="A", Ra2="A", log=False, forward=True):
    ''' Calculates random coil Hydrogen to Deuterium exchange rate for amide corresponding to side chain Raa
    @param Laa - amino acid letter code N-terminal to amide
    @param Raa - amino acid letter of amide
    @param pH - The pH of the experiment
    @param T - Temperature of the experiment

    Equation and constants derived from Bai, Englander (1980)
    ''' 

    if Raa=="P" or Raa=="CT" or Raa=="NT" or Laa=="NT":
        return 0
    # Constants
    pKD20c = 15.05
    pKcAsp = 4.48
    pKcGlu = 4.93
    pKcHis = 7.4
    R = 1.987
    EaA = 14000
    EaB = 17000
    EaW = 19000
    EaGlu = 1083
    EaHis = 7500
    if forward:
        EaAsp = 1000
        # the pD is different than the pH by +0.4 units
        pDcorr = pH+0.4
        ka = 0.694782306
        kb = 187003075.7
        kw = 0.000527046
    else:
        EaAsp = 960
        # the pD is different than the pH by +0.4 units
        pDcorr = pH
        ka = 0.4186477
        kb = 166666666.6
        kw = 0.0004186477

    inv_dTR = (1./T-1./293)/R

    FTa = numpy.exp(-1*EaA*inv_dTR)
    FTb = numpy.exp(-1*EaB*inv_dTR)
    FTw = numpy.exp(-1*EaW*inv_dTR)
    Dplus = 10**(-1*pDcorr)
    ODminus = 10**(pDcorr-pKD20c)

    # Get residue-specific effect factors

    L_ne = get_residue_neighbor_effects(Laa, pDcorr, T)
    R_ne = get_residue_neighbor_effects(Raa, pDcorr, T)

    Fa = L_ne[1]+R_ne[0]
    Fb = L_ne[3]+R_ne[2]

    if La2=="NT":
        Fa+=get_residue_neighbor_effects(La2, pDcorr, T)[1]
        Fb+=get_residue_neighbor_effects(La2, pDcorr, T)[3]
    if Ra2=="CT":
        Fa+=get_residue_neighbor_effects(Ra2, pDcorr, T)[0]
        Fb+=get_residue_neighbor_effects(Ra2, pDcorr, T)[2]

    Fa = 10**(Fa)
    Fb = 10**(Fb)

    krc = Fa*Dplus*ka*FTa + Fb*ODminus*kb*FTb + Fb*kw*FTw

    #print(Laa, Raa, "|", krc, L_ne[1], R_ne[0], L_ne[3], R_ne[2], "||", Fa, Fb, Fa*Dplus*ka*FTa, Fb*ODminus*kb*FTb, Fb*kw*FTw)

    return krc

def calculate_number_of_observable_amides(inseq, n_fastamides=2):
    #number of observable amides is equal to peptide length - 2, minus remaining prolines
    num_prolines = inseq.count('P', n_fastamides) + inseq.count('p', n_fastamides)
    return len(inseq) - num_prolines - n_fastamides

def get_sequence_intrinsic_rates(seq, pH, T, log=False, forward=True):
    i_rates = numpy.zeros(len(seq))
    i_rates[0] = calc_intrinsic_rate("NT", seq[0], pH, T, forward=forward)
    i_rates[1] = calc_intrinsic_rate(seq[0], seq[1], pH, T, La2="NT", forward=forward)
    for n in range(2, len(seq)-1):
        #print(n, seq[n],seq[n+1])
        L = seq[n-1]
        R = seq[n]
        i_rates[n] = calc_intrinsic_rate(L, R, pH, T, forward=forward)

    i_rates[-1] = calc_intrinsic_rate(seq[-2], seq[-1], pH, T, Ra2="CT", forward=forward)
    if log:
        # Suppress divide by zero error.
        with numpy.errstate(divide='ignore'):
            i_rates = numpy.log10(i_rates)

        #print("LOG", seq, i_rates)
        return i_rates
    else:
        return i_rates

def get_pdb_sequence(pdb_file, chain=None, offset=0):
    # Returns macromolecular sequence as a Pandas sequence
    seq = pd.Series()
    with open(pdb_file) as f:
        for l in f:
            #print l.split()
            line=l.split()
            if len(line) < 4:
                continue
            elif line[2]=="CA" and (chain is None or chain==line[4]):
                resnum=int(line[5])+offset
                #print resnum, line, len(self.get_fasta_sequence())
                if len(seq) == 0:
                    seq = pd.Series(ThreeToOne.get(line[3],'.'), index = [resnum])
                else:
                    seq = seq.append(pd.Series(ThreeToOne.get(line[3],'.'), index = [resnum]))
    #seq.index = seq.index + offset
    return seq


def calculate_peptide_mass(string):
    ''' Calculates the mass of a peptide string using
    only the most abundant isotopes.
    @param string - character string of one-letter amino acids

    Returns a float
    '''
    mass = 0
    for i in string:
        if i not in AA_list:
            raise Exception("tools.calculate_peptide_mass : unknown amino acid " + i + "")

        els = ResidueChemicalContent(i)

        ellist = ["C", "H", "O", "N", "S"]

        for i in range(len(els)):
            mass += els[i] * ElementMasses(ellist[i])[0][0]

    return mass

def calculate_peptide_chi(deuteration_by_time, exp, sigmas=None):
    # Given a set of deuterations at each timepoint,
    # as well as experimental results
    #
    # If timepoint sigmas are included, utilize those as chi sigmas.
    #
    return 0

def calculate_simple_deuterium_incorporation(rate, time):
    # Calculates the deuterium incorporation for a log(kex)
    # at time = t (seconds) assuming full saturation
    return 1 - math.exp(-1*(10**rate)*time)

def get_residue_deuteration_at_each_timepoint(dataset, protection_factors):
    # Returns the deuterium incorporation for each residue at each timepoint in
    # the given dataset and given protection factors
    timepoints = dataset.get_all_timepoints()
    deuterations_by_time = {}

    for tp in timepoints:
        deuterations = []
        for i in range(len(protection_factors)):
            if math.isnan(dataset.intrinsic[i]) or i == numpy.inf or i == -1 * numpy.inf:
                deuterations.append(0)
            else:
                log_kex = dataset.intrinsic[i] - protection_factors[i] 
                # Deuterium incorporation is scaled by the amount of deuterium in solution
                deut = calculate_simple_deuterium_incorporation(log_kex, tp.time) * dataset.conditions.saturation
                deuterations.append(deut)
        tp.set_deuteration(deut)
        deuterations_by_time[tp.time] = deuterations

    return deuterations_by_time

def calculate_incorporation(intrinsic, protection_factors, timepoints, offset=0):
    if len(intrinsic) != len(protection_factors):
        raise Exception("Intrinsic exchange factors and protection factor list not the same length")
    # incorporations is a dictionary of dictionaries, with the first dictionary keyed by residue number and the second 
    # It is indexed by residue number (1 = 1)
    #
    # res_incorps is keyed by timepoint
    incorporations = {}
    for n in range(len(intrinsic)):

        res_incorps = {}
        for tp in timepoints:
            if protection_factors[n] == -1:
                res_incorps[tp] = 0
            else:
                log_kex = intrinsic[n] - protection_factors[n]
                res_incorps[tp] = calculate_simple_deuterium_incorporation(log_kex, tp)
                #print("||||||||", intrinsic[n], protection_factors[n], log_kex, tp, res_incorps[tp])
        incorporations[n+1+offset] = res_incorps

    return incorporations

def get_peptide_deuteration(peptide, protection_factors):
    deuts = {}
    for tp in peptide.get_timepoints():
        deuts[tp.time] = get_timepoint_deuteration(peptide, tp.time, protection_factors)
    return deuts

def get_timepoint_deuteration(peptide, time, protection_factors):
    for r in peptide.get_observable_residue_numbers():
        pf = protection_factors[r-1]
        kr = peptide.dataset.intrinsic[r-1]
        deut += calculate_simple_deuterium_incorporation(kr - pf, time) * peptide.dataset.conditions.saturation
    return deut


def calc_peptide_isotopic_distribution(string, threshold = 0.1):
    ''' Calculates the natrual isotopic distribution of a peptide
    using the multinomial expansion
    @param string - character string of one-letter amino acids
    @param threshold - returned isotopes must be at least this percent of total

    Returns a list of tuples (mass, percent)

    Reference: Yergey (1983)
    '''
    for i in string:
        if i not in AA_list:
            raise Exception("tools.calculate_peptide_mass : unknown amino acid " + i + "")


def calculate_deut(rate, time):
    return 1-math.exp(-1*(10**rate)*time)

def simulate_peptides_data(seq, exch_rates, st_end_tups, timepoints, replicates=3, obs_error=5, percentD=True):

    peptides = []
    for i in st_end_tups:
        peptides.append(simulate_peptide_data(seq[i[0]-1:i[1]], i[0], timepoints, replicates, obs_error, percentD))

    return peptides


def simulate_peptide_data(seq, start_res, exch_rates, timepoints, replicates=3, obs_error=5, percentD=True):
    '''
    For a list of exchange rates, calculate simulated data including 
    gaussian noise at each timepoint.
    @param exch_rates - list of exchange rates for a set of contiguous residues
    @param timepoints - list of integer timepoints (in seconds)
    @param error - Standard deviation (in total D units)

    Returns a list of lists of 2D incorporation values.  
    '''
    peptide = system.Peptide(seq, start_res, start_res+len(seq))

    # Add data to the fragment
    for t in timepoints:
        tp = frag.add_timepoint(t)
        deut = 0
        for r in range(replicates):
            for i in exch_rates[2:]:
                # calc exch
                d = calculate_deut(i, tp.time)
                deut += d + normal(0, d*obs_error/100)

        tp.add_replicate(deut)

    return peptide


def simulate_state_data(state, rates, replicates, timepoints, pH=7, temp=283):
    return 0

def calculate_mass(string, percent_cutoff = 100):
    ''' Calculates the average mass of a peptide string
    @param string - AA sequence.  N and C-terms are implicitly added
    @param percent_cutoff - below this percentage, don't report the isotope (100 = only the major isotope)
    '''
    return 0

def subsequence_consistency(sequence, subsequence, start_residue):
    # Simple utility that compares a subsequence to a whole sequence
    # Returns True if 100% consistent. False otherwise
    #print(sequence, sequence[start_residue-1:start_residue-1+len(subsequence)])
    #print("DHSIUO", subsequence, start_residue, sequence)
    return sequence[start_residue-1:start_residue-1+len(subsequence)]==subsequence


import pyopenms as oms
import pandas as pd
import numpy as np



def ms_smoothing(spectrum):
    exp = oms.MSExperiment()
    gf = oms.GaussFilter()
    param = gf.getParameters()
    param.setValue("gaussian_width", 0.001)  # Adjust the width as needed
    gf.setParameters(param)

    exp.addSpectrum(spectrum)
    gf.filterExperiment(exp)
    spectrum = exp[0]  # Get back the modified spectrum

    return spectrum


def calculate_SN(spectrum: oms.MSSpectrum, mz_value: float, window: float = 1.5):
    # Extract peaks
    mz, intensity = spectrum.get_peaks()
    
    # Find the peak intensity at the mz_value
    peak_intensity = intensity[np.abs(mz - mz_value) < window].max()

    # Estimate the noise level as the mean intensity in the window
    noise_level = np.mean(intensity[np.abs(mz - mz_value) < window])
    
    # Calculate the signal-to-noise ratio
    sn_ratio = peak_intensity / noise_level
    
    return sn_ratio


from sklearn.covariance import EllipticEnvelope


def convolute_deuterated_distribution(non_deuterated, centroid_deuteration, sigma=1):

    # Creating a Gaussian distribution
    x = np.arange(len(non_deuterated) + centroid_deuteration)
    gaussian = np.exp(-(x - centroid_deuteration)**2 / (2 * sigma**2))
    gaussian /= np.sum(gaussian)
    
    # Convolving the non-deuterated distribution with the Gaussian
    convoluted = np.convolve(non_deuterated, gaussian,)

    
    return convoluted


def get_isotope_envelope(replicate, add_sn_ratio_to_peptide=True):
    #print(f'{replicate.peptide.sequence} {replicate.timepoint.time} {replicate.charge_state}')
    
    # Step 1: create the spectrum object
    seq = oms.AASequence.fromString(replicate.peptide.sequence) 
    mfull = seq.getMonoWeight()
    mz_values = replicate.raw_ms['m/z'].values - mfull/float(replicate.charge_state)
    intensities = replicate.raw_ms['Intensity'].values 
    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((mz_values, intensities))

    # Step 2: smoothing 
    spectrum = ms_smoothing(spectrum)

    # Step 3: Centroiding
    picker = oms.PeakPickerHiRes()
    picked_spectrum = oms.MSSpectrum()
    picker.pick(spectrum, picked_spectrum)

    mz_picked, intensity_picked = picked_spectrum.get_peaks()
    df_picked = pd.DataFrame({'m/z': mz_picked, 'Intensity': intensity_picked})

    # set the impossible peaks to 1e-10
    max_peak_num = np.sum(get_theoretical_isotope_distribution(replicate)['Intensity'] > 0.01) + replicate.peptide.num_observable_amides
    df_picked = df_picked[df_picked['m/z'] <= max_peak_num]

    # # if bad drop replicate
    if df_picked.empty:
        #replicate.timepoint.replicates.remove(replicate)
        #print(f'bad replicate droped: {replicate.peptide.sequence} {replicate.timepoint.time} {replicate.charge_state}')
        return None

    
    # Step 2: Select highest near integer
    max_mz = int(df_picked['m/z'].max())
    selected_mz = []
    selected_intensity = []
    sn_ratio = []
    
    for mz_int in range(max_mz+1):
        mask = (df_picked['m/z'] >= mz_int-0.1) & (df_picked['m/z'] <= mz_int+0.1)
        peaks_in_range = df_picked[mask]
        
        if not peaks_in_range.empty:
            max_peak = peaks_in_range.loc[peaks_in_range['Intensity'].idxmax()]
            selected_mz.append(max_peak['m/z'])
            selected_intensity.append(max_peak['Intensity'])
            #print(max_peak)
            sn_ratio.append(calculate_SN(spectrum, max_peak['m/z']))

    selected_df = pd.DataFrame({
        'm/z': selected_mz,
        'Intensity': selected_intensity,
        'sn_ratio': sn_ratio
    })

    # round m/z to integer
    selected_df['m/z'] = selected_df['m/z'].apply(round)

    # if bad drop replicate
    if selected_df.empty:
       #replicate.timepoint.replicates.remove(replicate)
       #print(f'bad replicate droped: {replicate.peptide.sequence} {replicate.timepoint.time} {replicate.charge_state}')
       return None

    # if no zero peak, add one
    if 0 not in selected_df['m/z'].values:
        df_0 = pd.DataFrame({'m/z': [0], 'Intensity': [0]})
        selected_df = pd.concat([df_0, selected_df], ignore_index=True).reset_index(drop=True)

    if add_sn_ratio_to_peptide:
        replicate.sn_ratio = selected_df['sn_ratio'].mean()

    selected_df['Intensity'] /= selected_df['Intensity'].sum()

    return selected_df

def filter_by_thoe_ms(replicate, add_avg_intensity=True):

    experimental_peaks = replicate.isotope_envelope.reshape(-1, 1)
    
    theoretical_peaks = convolute_deuterated_distribution(replicate.peptide.best_t0_replicate.isotope_envelope, replicate.deut*replicate.max_d/100)
    theoretical_peaks = theoretical_peaks.reshape(-1, 1)

    # # Fit the EllipticEnvelope to the theoretical data
    envelope = EllipticEnvelope(contamination=0.05)  # adjust contamination appropriately
    envelope.fit(theoretical_peaks)

    # # Predict the outliers in the experimental data
    experimental_outliers = envelope.predict(experimental_peaks)
    is_inlier = (experimental_outliers == 1)
    

    # replace the outlier with 1e-10

    replicate.isotope_envelope[~is_inlier] = 1e-10
    replicate.isotope_envelope /= replicate.isotope_envelope.sum()

    # print num
    #num_outliers = (~is_inlier).sum()
    #print(f'{num_outliers} outliers removed')



from pyopenms import AASequence, CoarseIsotopePatternGenerator

def get_theoretical_isotope_distribution(replicate):
    """
    return the mono charge state iso
    """    
    # Create an AASequence object from the peptide sequence
    peptide_obj = AASequence.fromString(replicate.peptide.sequence)

    # Get the empirical formula of the peptide
    formula = peptide_obj.getFormula()
    isotope_generator = CoarseIsotopePatternGenerator(10)
    isotope_distribution = isotope_generator.run(formula)

    # Get the monoisotopic mass
    mono_mz = peptide_obj.getMonoWeight() 

    # Adjust the m/z values relative to the monoisotopic mass and considering the charge state
    theo_mz = np.array([(iso.getMZ()) - mono_mz for iso in isotope_distribution.getContainer()])
    theo_intensity = np.array([iso.getIntensity() for iso in isotope_distribution.getContainer()])

    #plt.stem(theo_mz, theo_intensity, linefmt='-', markerfmt=' ', basefmt=" ",)
    df_theo = pd.DataFrame({'m/z': theo_mz, 'Intensity': theo_intensity})
    return df_theo


def refine_dataset(dataset):
    print('num of tps: ', len(dataset.get_all_timepoints()))
    print('num of reps: ', len([rep for tp in dataset.get_all_timepoints() for rep in tp.replicates]))

    # print('Refining dataset...')
    # for pep in dataset.peptides:
    #     for tp in pep.timepoints:
    #         tp.replicates = [rep for rep in tp.replicates if 
    #                          rep.raw_ms['Intensity'].max() >= 1e4 and
    #                          rep.sn_ratio >= 30 and
    #                          rep.max_d/rep.peptide.num_observable_amides >= 0.5]
                
    #         # Remove timepoints without replicates
    #         if tp.replicates == []:
    #             pep.timepoints.remove(tp)
    #             print(f'{pep.sequence} {tp.time} removed')
                
    #     # Remove peptides without timepoints or with no time 0
    #     if pep.timepoints == []:
    #         dataset.peptides.remove(pep)
    #         print(f'{pep.sequence} removed')


    # for pep in dataset.peptides:
    #     tp0_reps = [rep for tp in pep.timepoints for rep in tp.replicates if rep.timepoint.time == 0 ]
    #     if len(tp0_reps) == 0:
    #         dataset.peptides.remove(pep)
    #         print(f'{pep.sequence} removed')

    # print('num of tps after refining: ', len(dataset.get_all_timepoints()))


    # step2: calculate the average intensity for each timepoint
    for pep in dataset.peptides:
        pep.get_best_charge_state()

        # back exchange
        back_exchange = 1 - pep.max_d/pep.num_observable_amides
        pep.set_back_exchange(back_exchange)
        
    for tp in dataset.get_all_timepoints():
        sigma = get_iso_sigma(tp, loss='AE')
        tp.set_sigma(sigma)

    # for tp in dataset.get_all_timepoints():
    #     try:
    #         tp.calculate_avg_iso_envelope()
    #     except:
    #         print(tp.peptide.sequence, tp.time)
    #         raise ValueError('cannot calculate average iso envelope')
            

    return dataset




def set_t0_rep_score(dataset):
    '''
    Set the score of the all the 0s replicate, based on KL divergence 
    with theoretical isotope distribution
    '''
    t0_reps = [rep for pep in dataset.peptides for tp in pep.timepoints for rep in tp.replicates if rep.timepoint.time == 0 ]
    for t0_rep in t0_reps:
        divergence = get_divergence(t0_rep.isotope_envelope,
                                    get_theoretical_isotope_distribution(t0_rep)['Intensity'].values,
                                    method='JS')
        t0_rep.set_score(divergence)

    print("Set score for %d t0 replicates" % len(t0_reps))


def get_weighted_avg_iso_envelope(timepoint, weights=None):
    reps = timepoint.replicates
    
    weights = np.array([rep.raw_ms['Intensity'][rep.raw_ms['Intensity'] > 1e3].mean() for rep in reps])
    weights /= weights.sum()
    
    max_len = max(len(rep.isotope_envelope) for rep in reps)
    
    iso_envelopes = []
    for rep in reps:
        padded_envelope = custom_pad(rep.isotope_envelope, max_len)
        iso_envelopes.append(padded_envelope)
    
    iso_envelopes = np.array(iso_envelopes)
    
    weighted_average_envelope = np.average(iso_envelopes, axis=0, weights=weights)
    weighted_average_envelope /= weighted_average_envelope.sum()

    return weighted_average_envelope


def get_weighted_avg_iso_envelope(timepoint, weights=None):
    reps = timepoint.replicates

    max_len = max(len(rep.isotope_envelope) for rep in reps)
    
    iso_envelopes = []
    for rep in reps:
        padded_envelope = custom_pad(rep.isotope_envelope, max_len)
        iso_envelopes.append(padded_envelope)
    
    iso_envelopes = np.array(iso_envelopes)
    
    average_envelope = np.average(iso_envelopes, axis=0)
    average_envelope /= average_envelope.sum()

    return average_envelope


import itertools


def get_iso_sigma(timepoint, loss=None):
    reps = timepoint.replicates
    if loss is None:
        raise ValueError('need to specify loss')
    
    elif loss == 'AE':
        if len(reps) == 1:
            return 0.1
        else:
            rep_combinations = list(itertools.combinations(reps, 2))
            error =  np.average([get_sum_ae(com[0].isotope_envelope, com[1].isotope_envelope) for com in rep_combinations])
            
    elif loss == 'JSD':
        if len(reps) == 1:
            return 0.1
        else:
            rep_combinations = list(itertools.combinations(reps, 2))
            error =  np.average([get_divergence(com[0].isotope_envelope, com[1].isotope_envelope, method='JS') for com in rep_combinations])
    
    return error


from numba import njit

@njit
def custom_kl_divergence(p, q):
    """Calculate the Kullback-Leibler divergence between two probability distributions."""
        
    # Ensure the probabilities are normalized
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)

    divergence = np.sum(x1 * np.log(x1 / x2))
    return divergence

@njit
def jensen_shannon_divergence(p, q):
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)
    m = (x1 + x2) / 2
    return (custom_kl_divergence(x1, m) + custom_kl_divergence(x2, m)) / 2

@njit
def custom_pad(array, target_length, pad_value=1e-10):
    if len(array) >= target_length:
        padded_array = array[:target_length].astype(np.float64)
        return padded_array
    
    padded_array = np.full(target_length, pad_value, dtype=np.float64)
    
    for i in range(len(array)):
        if array[i] == 0:
            padded_array[i] = pad_value
        else:
            padded_array[i] = array[i]
    
    return padded_array


@njit
def norm_pad_two_arrays(p, q):
    '''
    normalize two arrays and pad them to the same size and 
    '''
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)

    size = max(len(x1), len(x2))
    x1 = custom_pad(x1, size)
    x2 = custom_pad(x2, size)
    return x1, x2


@njit
def get_divergence(p, q, method='KL'):
    
    x1, x2 = norm_pad_two_arrays(p, q)
    
    if method == 'KL':
        return custom_kl_divergence(x1, x2)
    elif method == 'JS':
        return jensen_shannon_divergence(x1, x2)
    

@njit
def get_mse(p, q):

    x1, x2 = norm_pad_two_arrays(p, q)
    
    return np.mean(np.square(x1 - x2))

@njit
def get_mae(p, q):

    x1, x2 = norm_pad_two_arrays(p, q)
    
    return np.mean(np.abs(x1 - x2))


@njit
def get_sum_ae(p, q):

    x1, x2 = norm_pad_two_arrays(p, q)
    return np.abs(x1 - x2).sum()

@njit
def event_probabilities(p_array):
    n = p_array.shape[0]
    # dp[i][j] stores the probability of j successes in the first i events
    dp = np.zeros((n+1, n+1))
    dp[0][0] = 1  # Base case: probability of 0 successes in 0 events is 1
    
    for i in range(1, n+1):
        p = p_array[i-1]
        q = 1 - p
        dp[i][0] = dp[i-1][0] * q  # Probability of 0 successes in i events
        
        for j in range(1, i+1):
            # Probability of j successes in i events
            dp[i][j] = dp[i-1][j-1] * p + dp[i-1][j] * q
    
    # The probabilities of 0, 1, 2, ..., n successes in n events
    probabilities = dp[n]
    
    # if negative values, set to 0
    if np.any(probabilities < 0):
        probabilities[probabilities < 0] = 0
        probabilities = probabilities / np.sum(probabilities)

    probabilities = custom_pad(probabilities, 20)

    return probabilities


def remove_reps_from_dataset(removing_reps, dataset):

    # remove the replicates from the dataset
    for pep in dataset.peptides:
        for tp in pep.timepoints:
            tp.replicates = [r for r in tp.replicates if r not in removing_reps]
            #print(f'{rep} removed')
                                 
            # Remove timepoints without replicates
            if tp.replicates == []:
                pep.timepoints = [t for t in pep.timepoints if t != tp]
                print(f'{pep.sequence} {tp.time} removed')
                
        # Remove peptides without timepoints or with no time 0
        if pep.timepoints == []:
            dataset.peptides = [p for p in dataset.peptides if p != pep]
            print(f'{pep.sequence} removed')

    for pep in dataset.peptides:
        tp0_reps = [rep for tp in pep.timepoints for rep in tp.replicates if rep.timepoint.time == 0 ]
        if len(tp0_reps) == 0:
            dataset.peptides = [p for p in dataset.peptides if p != pep]
            print(f'{pep.sequence} removed')
            
import random

def random_drop_reps(dataset, drop_pecent=20):
    all_reps = [rep for pep in dataset.peptides for tp in pep.timepoints for rep in tp.replicates if tp.time != 0]
    k = len(all_reps) * drop_pecent // 100
    droping_reps = random.sample(all_reps, k)

    remove_reps_from_dataset(droping_reps, dataset)
    return dataset