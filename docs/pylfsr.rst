Link to pylfsr documentation
---------------------------

**Example: 5 bit LFSR with x^5 + x^2 + 1**
~~~~~~~~~~~~~~~~~~~~~~~

::

  import numpy as np
  from spkit.pylfsr import LFSR
  
  L = LFSR()
  L.info()
  L.next()
  L.runKCycle(10)
  L.runFullCycle()
  L.info()
  tempseq = L.runKCycle(10000)    # generate 10000 bits from current state


`Check out full documentation of LFSR** <https://lfsr.readthedocs.io>`_
~~~~~~~~~~~~~~~~~

https://lfsr.readthedocs.io/
