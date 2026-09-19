# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'gpr_lib', 'gpr_lib.GP_prior', 'gpr_lib.Likelihood', 'gpr_lib.Utils',
        'model_learning', 'policy_learning', 'simulation_class',
        'robot_tossing_utils', 'sac_tossing',
    ],
    package_dir={'': 'python'},
)
setup(**setup_args)
