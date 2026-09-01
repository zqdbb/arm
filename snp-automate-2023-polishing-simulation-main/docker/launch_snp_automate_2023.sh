#!/bin/bash

source /opt/snp_automate_2023/install/setup.bash
ros2 launch snp_automate_2023 start.launch.xml sim_robot:=${SNP_SIM_ROBOT:-false} sim_vision:=${SNP_SIM_VISION:-false} bypass_execution:=${SNP_BYPASS_EXECUTION:-false}
