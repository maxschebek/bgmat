#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright 2025 Maximilian Schebek, Freie Universität Berlin
# Modified: 2025-10-03 - Adapted and extended for bgmat project. Further modifications should be documented here.


"""Utilities."""

from typing import Sequence

import chex
import haiku as hk

Array = chex.Array


class Parameter(hk.Module):
  """Helper Haiku module for defining model parameters."""

  def __init__(self,
               name: str,
               param_name: str,
               shape: Sequence[int],
               init: hk.initializers.Initializer):
    super().__init__(name=name)
    self._param = hk.get_parameter(param_name, shape=shape, init=init)

  def __call__(self) -> Array:
    return self._param

# Assuming 'params' is your parameter dict after init
def list_parameters(params):
    total = 0
    for module_name, module_params in params.items():
        for param_name, value in module_params.items():
            param_count = value.size
            print(f"{module_name}/{param_name}: {param_count}")
            total += param_count
    print(f"\nTotal parameters: {total}")
    return total

# Function to count the number of parameters
def count_parameters(params):
    def count_recursive(param_dict):
        total_params = 0
        for key, value in param_dict.items():
            if isinstance(value, dict):
                total_params += count_recursive(value)
            else:
                total_params += value.size
        return total_params

    return count_recursive(params)