"""Versions the optional sidecar envs must run (shared by install checks).

Lives in the runtime layer so both ``runtime.doctor`` and
``integrations.sleap`` can use it without an upward import.
"""

#: sleap 1.6.2 pulls sleap-nn 0.1.x. Newer sleap-nn (0.3.x) changed the data
#: utilities the shared-memory transport relies on and fails with an opaque
#: AttributeError in sleap_nn.data.utils mid-run. install.py pins 0.1.3.
SUPPORTED_SLEAP_NN_PREFIX = "0.1."
