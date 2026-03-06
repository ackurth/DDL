import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from place_field_formation import PlaceFieldFormation

if __name__ == "__main__":
    rng = np.random.default_rng()
    neuron_params = {
        "tau_m": 15.0,  # in ms
        "tau_syn": 5.0,  # in ms
        "tau_plast_short": 1000,
        "tau_plast_long": 3000,
        "alpha": 0.5,
        "beta": 5,
        "theta": 1,
        "pp_refac": 1000,  # ms
        "w_prox_max": 0.15,
        "w_prox_min": 0.0,
        'w_sum': 2,
        'delta_w': 0.0025,
        'shift': 300,
        'norm': False, 
        'dSpike_thres_high': 0.35,
        'dSpike_thres_low': 0.2,
        'prob_self_mediated': 0.75,
        'time_diff_self_mediated': 300
    }

    len_track = 300
    res = 1

    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_background": 0,  # in Hz
        "input_rate": 30,
        "len_track": 300, # in cm
        "velocity": 20, # in cm / s
    }

    num_neurons = 500

    num_inputs = 200

    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}
    neuron.create_place_field_centers()

    nums = 2
    time_steps = neuron._time_steps()


    PP_1 = np.random.choice(time_steps, num_neurons)

    repititions = []

    for i in range(nums):

        if i  == 0:
            state, t1 = neuron.run(
                learning=False, state=state, recording=True
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PP_1}
            )
        else:
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'self_gen': True}
            )
            print(traces['PP_onset'].sum())
        state, traces = neuron.run(
            learning=False, state=state, recording=True
        )


        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)

    results = []
    sorting = []
    for i in range(nums):

        traces = repititions[i]

        f_list = traces["f"]
        num_bins = len_track
        num_bins = len_track // 2

        time_window = time_steps // num_bins 

        reps = []

        for id in range(num_bins):
            rep = (
                np.mean(
                    f_list[:, id * time_window : (id + 1) * time_window],
                    axis=1,
                )
                * 50
            )
            reps.append(rep)


        res = np.asarray(reps)

        results.append(res)
        m = np.argsort(np.argmax(res, axis=0))
        sorting.append(m)

        repititions[i]['rate_avg'] = res
        repititions[i]['sorting'] = m

    with open('data/fig_2/reinforcement.pkl', 'wb') as f:
        pickle.dump(repititions, f)
    
    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}
    neuron.create_place_field_centers()
    nums = 2

    PP_2 = np.random.choice(time_steps, num_neurons)
    repititions = []

    for i in range(nums):

        if i  == 0:
            state, t1 = neuron.run(
                learning=False, state=state, recording=True
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PP_1}
            )

        else:
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PP_2}
            )
        state, traces = neuron.run(
            learning=False, state=state, recording=True
        )
        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)

    
    results = []
    sorting = []
    for i in range(nums):
        traces = repititions[i]

        f_list = traces["f"]

        num_bins = len_track
        num_bins = len_track // 2

        time_window = time_steps // num_bins 
        reps = []

        for id in range(num_bins):
            rep = (
                np.mean(
                    f_list[:, id * time_window : (id + 1) * time_window],
                    axis=1,
                )
                * 50
            )
            reps.append(rep)

        res = np.asarray(reps)

        results.append(res)
        m = np.argsort(np.argmax(res, axis=0))
        sorting.append(m)

        repititions[i]['rate_avg'] = res
        repititions[i]['sorting'] = m


    with open('data/fig_2/remapping.pkl', 'wb') as f:
        pickle.dump(repititions, f)

    import IPython
    IPython.embed()

