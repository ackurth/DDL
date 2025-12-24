import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel
from place_field_formation import PlaceFieldFormation

    
def full_activity(rep_id, repititions):
    t = repititions[rep_id]

    fig, axes = plt.subplots(10, 16, figsize=(40, 20))
    for i in range(10):
        for j in range(16):
            axes[i, j].set_ylim([0,50])
            axes[i, j].plot(50 * t['f'][j * 10 + i])
            PP_loc = t['PP_onset'][j * 10 + i]
            for time in np.where(PP_loc == 1)[0]:
                axes[i, j].axvline(x=time, c='red')

def fr_analysis(traces):
    f_list = traces["f"]
    pat_idx = traces["pat_idx"]
    pat_start_idx = traces["pat_start_idx"]
    width = traces["width"]

    reps = []
    for id, pat_start in zip(pat_idx, pat_start_idx):
        rep = np.mean(
            f_list[:, pat_start : pat_start + width],
            axis=1,
        )
        reps.append(rep)

    fr_mat = np.asarray(reps)
    sorting = np.argsort(np.argmax(fr_mat, axis=0))

    return fr_mat, sorting


if __name__ == "__main__":
    rng = np.random.default_rng()

    neuron_params = {
        "g_d_prox": 1.0,
        "g_d_dist": 1.0,
        "g_L": 1 / 15.0,
        "tau_m": 15.0,  # in ms
        "tau_syn": 5.0,  # in ms
        "tau_pp": 100.0,  # in ms
        "tau_rate": 200.0,  # in ms
        "tau_plast_short": 500,
        "tau_plast_long": 750,
        "lr": 1e-2,
        "tau_velo": 20.0,
        "alpha": 0.5,
        "beta": 5,
        "pp_refac": 3000,  # ms
        "up_cross_dur": 0,  # ms
        "w_prox_max": 0.5,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_PP": 3,
        "p_dSpike": 1 / 30,
        'w_sum': 2,
        'delta_w': 0.005,
        'dSpike_thres_high': 0.6,
        'dSpike_thres_low': 0.2
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 2,  # in Hz
        "mean_rate_dist": 2,  # in Hz
        "stim_dur": 200,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 40,
    }

    num_neurons = 160

    num_inputs = 500

    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}

    repititions = 1

    representations = []
    f = lambda x, mu, sigma: np.exp(-((x - mu) ** 2) / 2 / sigma**2)

    len_stim = 50
    min_shift = 5

    steps = num_inputs // min_shift

    inp_space = np.linspace(0, len_stim - 1, len_stim)
    inp = f(inp_space, len_stim // 2, len_stim // 4) * 1.5

    sim_params.update({"num_pats": steps})

    neuron.sim_params["num_pats"] = steps
    neuron.input_neurons = np.zeros(
        (neuron.sim_params["num_pats"], neuron.num_inp)
    )

    for i in range(steps):

        if len_stim + min_shift * i < num_inputs:
            neuron.input_neurons[i][
                min_shift * i : len_stim + min_shift * i
            ] = inp
        else:
            wrapped_idx = (len_stim + min_shift * i) % num_inputs
            neuron.input_neurons[i][min_shift * i : -1] = inp[
                : num_inputs - min_shift * i - 1
            ]
            neuron.input_neurons[i][:wrapped_idx] = inp[
                num_inputs - min_shift * i - 1 : -1
            ]
    
    neuron.input_neurons = np.roll(neuron.input_neurons, axis=1, shift=-len_stim // 2)

    repititions = []

    nums = 10

    for i in range(nums):

        if i  == 1:
            state, t1 = neuron.run(
                learning=False, state=state, recording=True, context_args={'context': False}
            )
            state, t1 = neuron.run(
                learning=True, state=state, recording=True, context_args={'context': True}
            )
        state, traces = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': False}
        )

        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)

    results = []
    sorting = []
    for i in range(nums):

        traces = repititions[i]

        f_list = traces["f"]
        pat_idx = traces["pat_idx"]
        pat_start_idx = traces["pat_start_idx"]
        width = int(sim_params['stim_dur'] / sim_params['dt'])

        reps = []
        for id, pat_start in zip(pat_idx, pat_start_idx):
            rep = (
                np.mean(
                    f_list[:, pat_start : pat_start + width],
                    axis=1,
                )
                * 50
            )
            reps.append(rep)

        res = np.asarray(reps)
        results.append(res)
        m = np.argsort(np.argmax(res, axis=0))
        sorting.append(m)

    rep_id = 0

    for rep_id in range(nums):
        full_activity(rep_id, repititions)

        plt.savefig(f'place_field_formation/full_{rep_id}.png')
        plt.close()
        
    #rep_id = 5
    #full_activity(rep_id, repititions)

    #plt.savefig(f'place_field_formation/full_{rep_id}.png')
    #
    #rep_id = 9
    #full_activity(rep_id, repititions)

    #plt.savefig(f'place_field_formation/full_{rep_id}.png')

    vmax = 50

    fig, axes = plt.subplots(nums, nums, figsize=(20, 20))

    for i in range(0, nums):
        for j in range(0, nums):
            if j > i:
                axes[i, j].remove()
            else:
                axes[i, j].imshow(results[i].T[sorting[j]], vmin=0, vmax=vmax)


    plt.savefig("place_field_formation/consolitdation.png")

    import IPython
    IPython.embed()

    repititions = []
    for i in range(2):
        state, traces = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': False}
        )
        if i == 0:
            state, t1 = neuron.run(
                learning=True, state=state, recording=True, context_args={'context': True}
            )
            state, t1 = neuron.run(
                learning=False, state=state, recording=True, context_args={'context': False}
            )
        state, traces = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': False}
        )

        repititions.append(traces)

    results = []
    sorting = []
    for i in range(2):

        traces = repititions[i]

        f_list = traces["f"]
        pat_idx = traces["pat_idx"]
        pat_start_idx = traces["pat_start_idx"]
        width = int(sim_params['stim_dur'] / sim_params['dt'])

        reps = []
        for id, pat_start in zip(pat_idx, pat_start_idx):
            rep = (
                np.mean(
                    f_list[:, pat_start : pat_start + width],
                    axis=1,
                )
                * 50
            )
            reps.append(rep)

        res = np.asarray(reps)
        results.append(res)

        m = np.argsort(np.argmax(res, axis=0))
        sorting.append(m)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(results[0].T[sorting[0]], vmin=0, vmax=vmax)
    axes[1, 0].imshow(results[1].T[sorting[0]], vmin=0, vmax=vmax)
    axes[1, 1].imshow(results[1].T[sorting[1]], vmin=0, vmax=vmax)

    axes[0, 1].remove()

    plt.savefig("place_field_formation/remapping.png")
