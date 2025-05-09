# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Experimenter that discretizes the parameters of search space."""

import copy
from typing import Sequence, Mapping

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


class DiscretizingExperimenter(experimenter.Experimenter):
  """DiscretizingExperimenter discretizes the parameters of search space."""

  def __init__(self,
               exptr: experimenter.Experimenter,
               discretization: Mapping[str, pyvizier.MonotypeParameterSequence],
               *,
               allow_oov: bool = False):
    """DiscretizingExperimenter discretizes continuous parameters.

    Currently only supports flat double search spaces. Note that the discretized
    parameters must fit within the bounds of the continuous parameters. This
    also supports CATEGORICAL parameters but feasible categories must be
    convertible to floats.

    Args:
      exptr: Underlying experimenter to be wrapped.
      discretization: Dict of parameter name to discrete/categorical values.
      allow_oov: Allows out of vocabulary values for Trial parameter in
        Evaluate. If True, evaluate the underlying experimenter at any given
        parameter values, whether feasible or not.

    Raises:
      ValueError: Non-double underlying parameters or discrete values OOB.
    """
    self._exptr = exptr
    self._discretization = discretization
    self._allow_oov = allow_oov
    exptr_problem_statement = exptr.problem_statement()

    if exptr_problem_statement.search_space.is_conditional:
      raise ValueError('Search space should not have conditional'
                       f' parameters  {exptr_problem_statement}')

    search_params = exptr_problem_statement.search_space.parameters
    param_names = [param.name for param in search_params]
    for name in discretization.keys():
      if name not in param_names:
        raise ValueError(f'Parameter {name} not in search space'
                         f' parameters for discretization: {search_params}')

    self._problem_statement = copy.deepcopy(exptr_problem_statement)
    self._problem_statement.search_space = pyvizier.SearchSpace()
    for parameter in search_params:
      if parameter.name not in discretization:
        self._problem_statement.search_space.add(parameter)
        continue

      if parameter.type != pyvizier.ParameterType.DOUBLE:
        raise ValueError(
            f'Non-double parameters cannot be discretized {parameter}')
      # Discretize the parameters.
      min_value, max_value = parameter.bounds
      for value in discretization[parameter.name]:
        float_value = float(value)
        if float_value > max_value or float_value < min_value:
          raise ValueError(f'Discretized values are not in bounds {parameter}')
      self._problem_statement.search_space.add(
          pyvizier.ParameterConfig.factory(
              name=parameter.name,
              feasible_values=discretization[parameter.name],
              scale_type=parameter.scale_type,
              external_type=parameter.external_type))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return self._problem_statement

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]) -> None:
    """Evaluate the trials after conversion to double."""

    old_parameters = []
    for suggestion in suggestions:
      old_parameters.append(suggestion.parameters)
      new_parameter_dict = {}
      for name, param in suggestion.parameters.items():
        if name in self._discretization:
          if self._allow_oov:
            if param.value not in self._discretization[name]:
              raise ValueError(
                  f'Parameter {param} not in {self._discretization[name]}')
          new_parameter_dict[name] = param.as_float
        else:
          new_parameter_dict[name] = param
      suggestion.parameters = pyvizier.ParameterDict(new_parameter_dict)
    self._exptr.evaluate(suggestions)
    for old_param, suggestion in zip(old_parameters, suggestions):
      suggestion.parameters = old_param

  def __repr__(self):
    return f'DiscretizingExperimenter({self._discretization}) on {self._exptr}'
