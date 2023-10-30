# !/usr/bin/python3
# -*- coding: utf-8 -*-

def run_simulation(bayesian_hdx_dir, sequence,outputdir, infile, offset=0, sigma0=1.0, saturation=1.0, percentD=False,
                   num_exp_bins=40, init="random", nsteps=10000, state_name=None):
    """Runs a simulation using the provided parameters."""

    import sys
    sys.path.append(bayesian_hdx_dir)

    import scoring, sampling, system, model, hxio, tools, analysis, plots

    # Initialize model
    sys = system.System(output_dir=outputdir, noclobber=False)
    mol = sys.add_macromolecule(sequence, "Test", initialize_apo=False)

    # Add states
    if state_name is None:
        raise ValueError("Must provide a state name")
    state = mol.add_state(f"{state_name}_1")
    state2 = mol.add_state(f"{state_name}_2")

    # Import data
    dataset = hxio.import_HXcolumns(infile,
                                    sequence,
                                    name="Data",
                                    percentD=percentD,
                                    conditions=None,
                                    error_estimate=sigma0,
                                    n_fastamides=2,
                                    offset=offset)

    # Add data to molecule state
    state.add_dataset(dataset)
    state2.add_dataset(dataset)

    sys.output.write_datasets()

    output_model = model.ResidueGridModel(state, grid_size=num_exp_bins)
    state.set_output_model(output_model)
    state2.set_output_model(output_model)

    sampler = sampling.MCSampler(sys, pct_moves=20, sigma_sample_level="timepoint")

    sys.output.initialize_output_model_file(state, output_model.pf_grids)
    sys.output.initialize_output_model_file(state2, output_model.pf_grids)

    sampler.run(nsteps, 10.0, write=True)

    pof = analysis.ParseOutputFile(outputdir + f"/models_scores_sigmas-{state_name}_1.dat")
    pof2 = analysis.ParseOutputFile(outputdir + f"/models_scores_sigmas-{state_name}_2.dat")
    pof.generate_datasets()
    pof2.generate_datasets()
    pof.calculate_random_sample_convergence()
    pof2.calculate_random_sample_convergence()

    conv = analysis.Convergence(pof, pof2)

    return conv.total_score_pvalue_and_cohensd(), conv.residue_pvalue_and_cohensd()


# Usage example
if __name__ == '__main__':

    bayesian_hdx_dir = '../../pyext/src'

    sequence = "IKKIGVLTSGGDAPGMNAAIRGVVRSALTEGLEVMGIYDGYLGLYEDRMVQLDRYSVSDMINRGGTFLGSARFPEFRDENIRAVAIENLKKRGIDALVVIGGDGSYMGAMRLTEMGFPCIGLPGTIDNDIKGTDYTIGFFTALSTVVEAIDRLRDTSSSHQRISVVEVMGRYCGDLTLAAAIAGGCEFVVVPEVEFSREDLVNEIKAGIAKGKKHAIVAITEHMCDVDELAHFIEKETGRETRATVLGHIQRGGSPVPYDRILASRMGAYAIDLLLAGYGGRCVGIQNEQLVHHDIIDAIENMKRPFKGDWLDCAEKMY"
    
    outputdir = "./output_test"
    
    infile = "./bayesian_hdx_ecpfk_APO.dat"
    res = run_simulation(bayesian_hdx_dir, sequence, outputdir, infile, state_name="bayesian_hdx_ecpfk_APO", nsteps=10000, num_exp_bins=40, init = "enumerate")
    print(res)


    infile = "./bayesian_hdx_ecpfk_ADP.dat"
    res = run_simulation(bayesian_hdx_dir, sequence, outputdir, infile, state_name="bayesian_hdx_ecpfk_ADP", nsteps=10000, num_exp_bins=40, init = "enumerate")
    print(res)


    infile = "./bayesian_hdx_ecpfk_PEP.dat"
    res = run_simulation(bayesian_hdx_dir, sequence, outputdir, infile, state_name="bayesian_hdx_ecpfk_PEP", nsteps=10000, num_exp_bins=40, init = "enumerate")
    print(res)


