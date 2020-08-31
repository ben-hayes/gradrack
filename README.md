# Gradrack

Gradrack is a library of differentiable, modular synthesis components intended
for constructing deep learning models in the style of Engel et al's DDSP
architecture [1]. Its focus is on providing music-specific synthesis modules
implemented using vectorisable PyTorch operations to enable GPU accelerated
synthesis and processing across batches.

## Structure

Gradrack provides simple components that can be used independently, or combined
together to form more complex synthesisers. Currently the following components
are implemented:

* Oscillator Base
* Sinusoidal Oscillator
* ADSR Envelope

and the following synthesiser containers are implemented:

* FM Synth

## Installation

Gradrack can be installed by cloning this repo and running:

```bash
pip install -e .
```

or through `pip` using:

```bash
pip install gradrack
```

However, as Gradrack is in early development, no guarantees are made about the
state of the `pip` release.

## Usage

### Oscillators 

A simple oscillator can be created as follows:

```python
import torch

from gradrack.oscillators import Sinusoid

osc = Sinusoid()
```

And a signal can be generated as follows:

```python

frequency = torch.Tensor([440.0])

x = osc(frequency, length=1000, sample_rate=44100)
```

More complex input shapes can also be used. This can be useful, for example, to
render audio across multiple batches simultaneously. In the following example,
we use a frequency input of shape (64, 1), and a phase modulation input of shape
(1000,). Gradrack interprets these input shapes to generate an output of shape
(64, 1000) -- it has computed 64 separate signals of length 1000 in parallel:

```python
frequency = torch.rand(64, 1) * 900 + 100  # 64 freqs from 100 Hz to 1000 Hz
phase_mod = torch.linspace(0, 1, 1000)  # 1000 sample ramp from 0.0 to 1.0

x = osc(frequency, phase_mod, sample_rate=44100)
```

Similarly, an envelope generator can be easily constructed:

```python
from gradrack.generators import ADSR

eg = ADSR()
```


### Generators

Envelope generators require a gate signal to generate output. This is a signal
that moves from zero to one at the start of a note event, and from one to zero
at the end. This input representation allows similar flexibility to that seen
in Eurorack modular synthesisers, and allows the envelope generator to retrigger
itself multiple times in a single generation.

Gate signals can be easily created by concatenating zeros and ones:

```python
gate = torch.cat((torch.ones(44100), torch.zeros(44100)))

env = eg(gate, 0.2, 0.3, 0.4, 0.5, sample_rate=44100)
```

### Synths

Gradrack also provides synthesiser containers. These implement routing logic
whilst being composable with other Gradrack components.

An FM synthesiser can therefore be constructed as follows:

```python
from gradrack.synths import FMSynth

oscs = Sinusoid(), Sinusoid()
egs = ADSR(), ADSR()
modulation_routing = ((1, 0),)  # route operator 1's output to operator 0's
                                # phase mod input

syn = FMSynth(oscs, egs, ((1, 0)), sample_rate=44100)
```

Generating audio requires any parameters the underlying components would need
to perform their function. Therefore:

```python
gate = torch.cat((torch.ones(22050), torch.zeros(22050)))
frequency = torch.Tensor([300.])
ratios = (1.0, 2.0)  # operator tuning ratios
eg_params = (
    (0.2, 0.4, 0.1, 0.4),  # ADSR parameters
    (0.1, 0.5, 0.3, 0.7)
)
operator_gains = (1.0, 0.7)

x = syn(gate, frequency, ratios, eg_params, operator_gains)
```

## References

[1] J. Engel, L. (Hanoi) Hantrakul, C. Gu, and A. Roberts, ‘DDSP: Differentiable
Digital Signal Processing’, presented at the International Conference on
Learning Representations, 2020, Accessed: Jan. 29, 2020. [Online]. Available:
https://openreview.net/forum?id=B1x1ma4tDr.


