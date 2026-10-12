"""Versions the optional sidecar envs must run (shared by install checks).

Lives in the runtime layer so both ``runtime.doctor`` and
``integrations.sleap`` can use it without an upward import.
"""

#: install.py pins sleap 1.6.5 + sleap-nn 0.3.3. Envs built from the earlier
#: pin (sleap 1.6.2 + sleap-nn 0.1.3) give byte-identical pose output, so they
#: stay supported. 0.2.x was never verified; 0.4 may drop the legacy
#: ``sleap_nn.inference.predictors`` module the pose service uses.
SUPPORTED_SLEAP_NN_PREFIXES = ("0.3.", "0.1.")

#: Human-readable form of SUPPORTED_SLEAP_NN_PREFIXES for error messages.
SUPPORTED_SLEAP_NN_TEXT = "0.3.x (or 0.1.x)"

#: The sleap release install.py pins; quoted in fix-it messages.
SLEAP_PIN = "1.6.5"
