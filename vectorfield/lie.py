import os, sys
sys.path.append(os.path.dirname(__file__))

try:
    from _vectorfield import (SE3, se3, SO31, so31)
except ImportError:
    raise ImportError("The vectorfield module requires the _vectorfield C extension module to be built.")