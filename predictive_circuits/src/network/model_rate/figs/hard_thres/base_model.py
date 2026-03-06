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
        }

        self.shift_steps = int(self.neuron_params['shift'] / self.sim_params["dt"])

        self.rng = rng

    def _firing_rate(self, V, theta=1):
        return 1.0 / (
            1.0
            + self.neuron_params["alpha"]
            * np.exp(-self.neuron_params["beta"] * (V - theta))
        )
    
    def _weight_norm(self, w_prox):
        return w_prox / (w_prox.sum(axis=1)[:, np.newaxis] / self.neuron_params['w_sum'])
    
    def _time_steps(self):
        pass

    def _input(self):
        pass

    def _process_misc(self, misc, i):
        pass
    
    def _process_misc_after(self, misc_after):
        pass

    def _PP_generation(self, context_args):
        pass

    def _init_state(self, state):
        '''
        Initialize model.
        If
            state dictionary contains state variable -> use passed state variable
        Else
            set state variable to default value
        '''

        # Neuron variables
        self.V = state.get("V", np.zeros(self.num_neurons)) # membrane potential
        self.I_syn = state.get("I_syn", np.zeros(self.num_inp)) # input caused by incoming spikes
        self.PSP = state.get("PSP", np.zeros(self.num_inp)) # evoked PSPs 
        self.PPs = state.get("PPs", np.zeros(self.num_neurons, dtype=bool)) # plateau potentials
        self.PP_onset = state.get("PP_onset", np.zeros(self.num_neurons, dtype=bool)) # plateau potential onset
        self.spike_buffer = state.get("spike_buffer", np.zeros((self.num_inp, self.shift_steps), dtype=bool)) # spike buffer

        self.mask_cross_low = np.zeros(self.num_neurons, dtype=bool)
        self.mask_cross_high = np.zeros(self.num_neurons, dtype=bool)
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

        self.trace_pre = state.get("trace_pre", np.zeros(self.num_inp))
        self.trace_post = state.get("trace_post", np.zeros(self.num_neurons))

        self.trace_pre_2 = state.get("trace_pre_2", np.zeros(self.num_inp))
        self.trace_post_2 = state.get("trace_post_2", np.zeros(self.num_neurons))

        # Fetch of alternative initialize proximal feedforward and recurrent weights

        try:
            self.w_prox = state["w_prox"]
        except:
            self.w_prox = self.rng.normal(2 / self.num_inp,
                                          2 / self.num_inp / 2,
                                          size=(self.num_neurons, self.num_inp))
            self.w_prox[self.w_prox < 0] = 0
            self.w_prox = self._weight_norm(self.w_prox)

        self.w_rec = 0.01 * (-1 * np.ones((self.num_neurons, self.num_neurons)) + np.eye(self.num_neurons))


    def _generate_data_container(self, time_steps):
        self.data_container = {
            "f": np.zeros((self.num_neurons, time_steps)),
            "cross_low": np.zeros((self.num_neurons, time_steps)),
            "cross_high": np.zeros((self.num_neurons, time_steps)),
            "spks": np.zeros((self.num_neurons, time_steps)),
            "V": np.zeros((self.num_neurons, time_steps)),
            "PP_onset": np.zeros((self.num_neurons, time_steps)),
            "trace_pre": np.zeros((self.num_inp, time_steps)),
            "trace_post": np.zeros((self.num_neurons, time_steps)),
            "trace_pre_2": np.zeros((self.num_inp, time_steps)),
            "trace_post_2": np.zeros((self.num_neurons, time_steps)),
            "spike_vec": np.zeros((self.num_inp, time_steps)),
            "pat_start_idx": [],
            "pat_idx": [],
        }

    def _record(self, i):
        self.data_container["f"][:, i] = self.f.copy()
        self.data_container["V"][:, i] = self.V.copy()
        self.data_container["PP_onset"][:, i] = self.PP_onset.copy()
        self.data_container["cross_low"][:, i] = self.cross_low.copy()
        self.data_container["cross_high"][:, i] = self.cross_high.copy()
        self.data_container["trace_pre"][:, i] = self.trace_pre.copy()
        self.data_container["trace_post"][:, i] = self.trace_post.copy()
        self.data_container["trace_pre_2"][:, i] = self.trace_pre_2.copy()
        self.data_container["trace_post_2"][:, i] = self.trace_post_2.copy()

    def run(
        self, state={}, learning=True, recording=False, context_args={}, input_params={}, misc_after={}
    ):

        self.learning = learning
        self.recording = recording

        self._init_state(state)
        self.pp_refac_counter *= 0

        if recording:
            self._generate_data_container(self._time_steps(**input_params))

        up_cross = np.zeros(self.num_neurons, dtype=bool)
        down_cross = np.zeros(self.num_neurons, dtype=bool)
        
        self.cross_low = np.zeros(self.num_neurons)
        self.cross_high = np.zeros(self.num_neurons)
        self.f = np.zeros(self.num_neurons)

        for (i, p_rand_prox, misc) in tqdm(self._input(**input_params), desc="[running]"):
            self.spike_buffer = np.roll(self.spike_buffer, axis=1, shift=1)

            self.PPs *= False

            # Sample input
            spike_vec = self.rng.random(self.num_inp) < p_rand_prox
            self.spike_buffer[:, 0] = spike_vec

            self.I_syn = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_syn"]
            ) * self.I_syn
            self.I_syn[spike_vec] += 1.0 / (
                self.neuron_params["tau_m"] * self.neuron_params["tau_syn"]
            )

            # Neuron potentials
            self.PSP = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_m"]
            ) * self.PSP + self.I_syn * self.sim_params["dt"]
            self.PSP_unit_prox = self.PSP * 25.0

            # Membrane potential cause by proximal inputs
            V_prev = self.V.copy()
            self.V = self.w_prox @ self.PSP_unit_prox + self.w_rec @ self.f

            # Somatic firing and dendritic prediction
            f_prev = self._firing_rate(V_prev, theta=self.neuron_params['theta'])
            self.f = self._firing_rate(self.V, theta=self.neuron_params['theta'])

            # dSpike
            self.mask_cross_low = ((f_prev < self.neuron_params['dSpike_thres_low'])
                                   * (self.neuron_params['dSpike_thres_low'] < self.f))

            self.mask_cross_high = ((f_prev < self.neuron_params['dSpike_thres_high'])
                                    * (self.neuron_params['dSpike_thres_high'] < self.f))

            self.cross_low[self.mask_cross_low] = i
            self.cross_high[self.mask_cross_high] = i
            
            self._PP_generation(i, **context_args)

            mask = self.PPs * (self.pp_refac_counter == 0)
            self.PP_onset = mask

            self.pp_refac_counter[self.PP_onset] += (
                self.aux_params["pp_refac_width"]
            )

            self.trace_pre *= np.exp(- self.sim_params["dt"] / self.neuron_params["tau_plast_short"])
            self.trace_pre[spike_vec] += 1.0
            
            self.trace_pre_2 *= np.exp(- self.sim_params["dt"] / self.neuron_params["tau_plast_long"])
            self.trace_pre_2[spike_vec] += 1.0

            self.trace_post *= np.exp(- self.sim_params["dt"] / self.neuron_params["tau_plast_short"])
            self.trace_post[mask] += 1.0
            self.trace_post_2 *= np.exp(- self.sim_params["dt"] / self.neuron_params["tau_plast_long"])
            self.trace_post_2[mask] += 1.0

            if learning:
                self.w_prox[:, spike_vec] += self.neuron_params['delta_w'] * (
                    self.trace_post[:, np.newaxis]
                    - np.minimum((self.w_prox[:, spike_vec] / self.neuron_params["w_prox_max"]) ** 0.5, 0.7) * 1.1
                    * self.trace_post_2[:, np.newaxis]
                )

                self.w_prox[mask, :] += self.neuron_params['delta_w'] * (
                    self.trace_pre[np.newaxis, :]
                    - np.minimum((self.w_prox[mask, :] / self.neuron_params["w_prox_max"]) ** 0.5, 0.7) * 1.1
                    * self.trace_pre_2[np.newaxis, :])

                self.w_prox[self.w_prox > self.neuron_params["w_prox_max"]] = (
                    self.neuron_params["w_prox_max"]
                )
                self.w_prox[self.w_prox < self.neuron_params["w_prox_min"]] = (
                    self.neuron_params["w_prox_min"]
                )
            
                if self.neuron_params['norm']:
                    self.w_prox = self._weight_norm(self.w_prox)

            self.pp_refac_counter[self.pp_refac_counter > 0] -= 1
            self.up_cross_counter[self.up_cross_counter > 0] -= 1

            if self.neuron_params['norm']:
                self.w_prox = self._weight_norm(self.w_prox)

            if recording:
                self._record(i)


            self._process_misc(misc, i)


        state = {
            "I_syn": self.I_syn.copy(),
            "PSP": self.PSP.copy(),
            "V": self.V.copy(),
            "spike_buffer": self.spike_buffer.copy(),
            "up_cross_counter": self.up_cross_counter.copy(),
            "pp_refac_counter": self.pp_refac_counter.copy(),
            "w_prox": self.w_prox.copy(),
            "trace_post": self.trace_post.copy(),
            "trace_pre": self.trace_pre.copy(),
            "trace_post_2": self.trace_post.copy(),
            "trace_pre_2": self.trace_pre.copy(),
        }

        self._process_misc_after(misc_after=misc_after)

        if recording:
            return state, self.data_container

        else:
            return state
