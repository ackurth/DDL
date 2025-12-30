import numpy as np
from tqdm import tqdm

class BaseModel:

    def __init__(
        self, num_neurons, num_inp, neuron_params, sim_params, rng
    ):

        self.num_neurons = num_neurons 

        self.num_inp = num_inp

        self.neuron_params = neuron_params
        self.sim_params = sim_params
        self.aux_params = {
            "pp_refac_width": int(
                self.neuron_params["pp_refac"] / self.sim_params["dt"]
            ),
            "up_cross_dur_width": int(
                self.neuron_params["up_cross_dur"] / self.sim_params["dt"]
            ),
        }

        self.rng = rng
    def _firing_rate(self, V, theta=1):
        return 1.0 / (
            1.0
            + self.neuron_params["alpha"]
            * np.exp(-self.neuron_params["beta"] * (V - theta))
        )

    def init_state(self, state):
        self.I_syn_prox = state.get("I_syn_prox", np.zeros(self.num_inp))

        self.PSP_prox = state.get("PSP_prox", np.zeros(self.num_inp))

        self.V_prox = state.get("V_prox", np.zeros(self.num_neurons))

        self.PPs = state.get("PPs", np.zeros(self.num_neurons, dtype=bool))
        
        self.PP_onset = state.get("PP_onset", np.zeros(self.num_neurons, dtype=bool))

        self.mask_cross_low = state.get("mask_cross_low", np.zeros(self.num_neurons, dtype=bool))
        self.mask_cross_high = state.get("mask_cross_high", np.zeros(self.num_neurons, dtype=bool))

        self.cross_low = np.zeros(self.num_neurons)
        self.cross_high = np.zeros(self.num_neurons)

        self.pp_refac_counter = state.get(
            "pp_refac_counter", np.zeros(self.num_neurons)
        )
        self.up_cross_counter= state.get(
            "up_cross_counter", np.zeros(self.num_neurons)
        )
        self.theta = state.get(
            "theta", self.neuron_params["theta"] * np.ones(self.num_neurons)
        )

        self.ET_1 = state.get("ET_1", np.zeros(self.num_inp))
        self.ET_2 = state.get("ET_2", np.zeros(self.num_inp))
        self.IS_1 = state.get("IS_1", np.zeros(self.num_neurons))
        self.IS_2 = state.get("IS_2", np.zeros(self.num_neurons))

        # Fetch of alternative initialize proximal feedforward and recurrent weights

        self.w_prox = state.get(
            "w_prox",
            (
                #np.abs(
                #    self.rng.standard_normal(
                #        size=(self.num_neurons, self.num_inp)
                #    )
                #)
                #/ self.num_inp

                self.rng.normal(self.neuron_params['w_sum'] / self.num_inp,
                                self.neuron_params['w_sum'] / self.num_inp / 2,
                                size=(self.num_neurons, self.num_inp)
                                )


            ),
        )
        self.w_prox[self.w_prox < 0] = 0
        if self.neuron_params['norm']:
            self.w_prox /= self.w_prox.sum(axis=1)[:, np.newaxis] / self.neuron_params['w_sum']

    def generate_container(self, time_steps):
        self.container = {
            "f": np.zeros((self.num_neurons, time_steps)),
            "spks": np.zeros((self.num_neurons, time_steps)),
            "V_prox": np.zeros((self.num_neurons, time_steps)),
            "PP_onset": np.zeros((self.num_neurons, time_steps)),
            "pat_start_idx": [],
            "pat_idx": [],
        }

    def time_steps(self):
        pass

    def input(self):
        pass

    def process_misc(self, misc, i):
        pass

    def PP_generation(self, context_args):
        pass

    def record(self, i):
        self.container["f"][:, i] = self.f
        self.container["V_prox"][:, i] = self.V_prox
        self.container["PP_onset"][:, i] = self.PP_onset

    def run(
        self, state={}, learning=True, recording=False, context_args={}, input_params={}
    ):


        self.learning = learning
        self.recording = recording

        self.init_state(state)
        self.pp_refac_counter *= 0

        if recording:
            self.generate_container(self.time_steps(**input_params))

        up_cross = np.zeros(self.num_neurons, dtype=bool)
        down_cross = np.zeros(self.num_neurons, dtype=bool)

        for (i, p_rand_prox, misc) in tqdm(self.input(**input_params), desc="[running]"):

            self.PPs *= False

            self.process_misc(misc, i)

            # Sample input
            spike_vec_prox = self.rng.random(self.num_inp) < p_rand_prox

            self.I_syn_prox = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_syn"]
            ) * self.I_syn_prox
            self.I_syn_prox[spike_vec_prox] += 1.0 / (
                self.neuron_params["tau_m"] * self.neuron_params["tau_syn"]
            )

            self.ET_1 = (
                1.0
                - self.sim_params["dt"] / self.neuron_params["tau_plast_short"]
            ) * self.ET_1
            self.ET_1[spike_vec_prox] += 1.0
            self.ET_2 = (
                1.0
                - self.sim_params["dt"] / self.neuron_params["tau_plast_long"]
            ) * self.ET_2
            self.ET_2[spike_vec_prox] += 1.0

            # Neuron potentials
            self.PSP_prox = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_m"]
            ) * self.PSP_prox + self.I_syn_prox * self.sim_params["dt"]
            self.PSP_unit_prox = self.PSP_prox * 25.0

            # Membrane potential cause by proximal inputs
            V_prox_prev = self.V_prox.copy()
            self.V_prox = self.w_prox @ self.PSP_unit_prox

            # Somatic firing and dendritic prediction
            f_prev = self._firing_rate(V_prox_prev, theta=self.neuron_params['theta'])
            self.f = self._firing_rate(self.V_prox, theta=self.neuron_params['theta'])

            # dSpike
            self.mask_cross_low = ((f_prev < self.neuron_params['dSpike_thres_low'])
                                   * (self.neuron_params['dSpike_thres_low'] < self.f)
                                   * ~self.mask_cross_low)

            self.mask_cross_high = ((f_prev < self.neuron_params['dSpike_thres_high'])
                                    * (self.neuron_params['dSpike_thres_high'] < self.f)
                                    * ~self.mask_cross_high)

            self.cross_low[self.mask_cross_low] = i
            self.cross_high[self.mask_cross_high] = i

            self.PP_generation(i, **context_args)

            mask = self.PPs * (self.pp_refac_counter == 0)

            self.PP_onset = mask

            self.pp_refac_counter[mask] += (
                self.aux_params["pp_refac_width"]
            )

            self.IS_1 = (
                1.0
                - self.sim_params["dt"] / self.neuron_params["tau_plast_short"]
            ) * self.IS_1
            self.IS_1[mask] += 1.0
            self.IS_2 = (
                1.0
                - self.sim_params["dt"] / self.neuron_params["tau_plast_long"]
            ) * self.IS_2
            self.IS_2[mask] += 1.0

            if learning:
                self.w_prox[:, spike_vec_prox] += (
                    self.neuron_params['delta_w']
                    * np.tanh(self.IS_1[:, np.newaxis])
                    - self.neuron_params['delta_w'] 
                    * self.w_prox[:, spike_vec_prox]
                    / self.neuron_params["w_prox_max"]
                    * np.tanh(self.IS_2[:, np.newaxis])
                )

                self.w_prox[mask, :] += (
                    self.neuron_params['delta_w']
                    * np.tanh(self.ET_1[np.newaxis, :])
                    - self.neuron_params['delta_w'] 
                    * self.w_prox[mask, :]
                    / self.neuron_params["w_prox_max"]
                    * np.tanh(self.ET_2[np.newaxis, :])
                )

                self.w_prox[self.w_prox > self.neuron_params["w_prox_max"]] = (
                    self.neuron_params["w_prox_max"]
                )
                self.w_prox[self.w_prox < self.neuron_params["w_prox_min"]] = (
                    self.neuron_params["w_prox_min"]
                )
            
            if self.neuron_params['norm']:
                self.w_prox /= self.w_prox.sum(axis=1)[:, np.newaxis] / self.neuron_params['w_sum'] 

            self.pp_refac_counter[self.pp_refac_counter > 0] -= 1
            self.up_cross_counter[self.up_cross_counter > 0] -= 1

            if recording:
                self.record(i)

        state = {
            "I_syn_prox": self.I_syn_prox,
            "PSP_prox": self.PSP_prox,
            "V}}_prox": self.V_prox,
            "up_cross_counter": self.up_cross_counter,
            "pp_refac_counter": self.pp_refac_counter,
            "w_prox": self.w_prox,
            "IS_1": self.IS_1,
            "IS_2": self.IS_2,
            "ET_1": self.ET_1,
            "ET_2": self.ET_2,
        }

        if recording:
            return state, self.container

        else:
            return state
