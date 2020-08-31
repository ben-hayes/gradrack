import torch


class FMSynth(torch.nn.Module):
    """A PyTorch module representing an FM synthesiser.
    
    A PyTorch module that can be composed with Gradrack oscillators and
    envelope generators to create a differentiable frequency modulation
    synthesiser. Routing of operators is set on construction and this routing
    graph is resolved to generate samples.
    """    
    def __init__(
        self,
        operators,
        envelope_generators,
        operator_routing,
        sample_rate=None,
    ):
        """Constructs an FM synth

        Args:
            operators (tuple): A tuple of Gradrack oscillators
            envelope_generators (tuple): A tuple of Gradrack envelope generators
            operator_routing (tuple): A tuple of modulator-carrier index pairs
                defining the modulator routing graph.
            sample_rate (float, optional): The sample rate in Hz. Defaults to 
                None.
        """    
        super().__init__()
        self.operators = operators
        self.envelope_generators = envelope_generators
        self.sample_rate = sample_rate

        self.operator_modulation_sources = [[] for _ in self.operators]
        for routing in operator_routing:
            self.operator_modulation_sources[routing[1]].append(routing[0])

        self.terminal_operators = self._find_terminal_operators(
            self.operator_modulation_sources
        )

    def forward(self, gate, frequency, ratios, eg_params, operator_gains=None):
        """Use the FM synth to generate a sound.
        
        Generate a signal given gate, fundamental frequency, and tuning ratios, 
        alongside envelope generator parameters and operator gains. Resolves the
        modulation graph recursively whilst memoising calls to each operator to
        save on computation time.

        Args:
            gate (torch.Tensor): A tensor representing the gate signal. Of the
                same format accepted by gradrack.generators.ADSR
            frequency (torch.Tensor): A tensor depicting the fundamental 
                frequency of each batch. Can vary over time.
            ratios (tuple): A tuple of operator tuning ratios in relation to 
                the fundamental frequency.
            eg_params (tuple): A tuple of tuples, each of which is exploded as
                positional arguments after gate, into the forward call of the
                envelope generator of the corresponding index.
            operator_gains (tuple, optional): A tuple of operator gains. 
                Defaults to None.

        Returns:
            torch.Tensor: The generated signal
        """        
        operator_outputs = [None] * len(self.operators)
        envelopes = []

        if frequency.shape[-1] == 1:
            frequency = frequency.repeat_interleave(gate.shape[-1], dim=-1)

        for n, eg in enumerate(self.envelope_generators):
            envelopes.append(eg(gate, *eg_params[n], self.sample_rate))

        def render_operator(n):
            if operator_outputs[n] is None:
                mod_sources = [
                    render_operator(m)
                    for m in self.operator_modulation_sources[n]
                ]

                if len(mod_sources) > 0:
                    phase_mod = torch.stack(mod_sources, dim=0).sum(dim=0)
                else:
                    phase_mod = None

                if operator_gains is not None:
                    gain = operator_gains[n]
                else:
                    gain = 1.0

                operator_outputs[n] = (
                    self.operators[n](
                        frequency * ratios[n],
                        sample_rate=self.sample_rate,
                        phase_mod=phase_mod,
                    )
                    * envelopes[n]
                    * gain
                )

            return operator_outputs[n]

        output_signal = torch.stack(
            [
                render_operator(terminal_operator)
                for terminal_operator in self.terminal_operators
            ],
            dim=0,
        ).sum(dim=0)

        return output_signal

    def _find_terminal_operators(self, operator_modulation_sources):
        """Resolve the routing graph to find pure-carrier operators. These are
        our output.        
        """        
        terminal_operators = []
        for i, _ in enumerate(operator_modulation_sources):
            is_terminal_operator = True
            for j, _ in enumerate(operator_modulation_sources):
                if i == j:
                    continue

                if i in operator_modulation_sources[j]:
                    is_terminal_operator = False
                    break

            if is_terminal_operator:
                terminal_operators.append(i)

        return terminal_operators
