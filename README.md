# differentiable-adaptive-optics
Source code and supplementary materials for the publication

## Contents

Source code is located in `src/`
Example notebooks in `notebooks/`

## Installation

Code was tested with **Python 3.9**. Earlier versions should also work, but **Numpy 1.20** is was used as it is the first release supporting native `np.typing`. 

1. Clone the repository
```bash
git clone https://github.com/ivan-vishniakou/differentiable-adaptive-optics.git
```
2. Navigate to  the cloned repository and install
```bash
cd differentiable-adaptive-optics
pip install .
```
3. Make sure the requirements are satisfied form `requiremets.txt`, either manually or by running
```bash
pip install -r requirements.txt
```


## Usage
Run the notebooks or use the code by importing the ```diffao``` package
```python
import numpy as np
from diffao.czt_tf import czt, czt_factors

RES = 512

signal = np.zeros(RES)
signal[RES//10:-RES//10] = 1.0

cf = czt_factors(RES, RES)
transformed = czt(signal, cf)