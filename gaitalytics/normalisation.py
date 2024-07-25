"""This module provides classes for batch normalisation of gait data in a trial."""

from abc import ABC, abstractmethod

import gaitalytics.model as model


class BaseNormaliser(ABC):
    """Base class for normalisers.

    This class provides a common interface for normalising data.
    """

    @abstractmethod
    def normalise(
        self, trial: model.Trial | model.TrialCycles
    ) -> model.Trial | model.TrialCycles:
        """Normalises the input data.

        Args:
            trial: The trial to be normalised.

        Returns:
            model.Trial: A new trial containing the normalised data.
            model.TrialCycles: A new segmented trial containing the normalised data
        """
        raise NotImplementedError


class LinearTimeNormaliser(BaseNormaliser):
    """A class for normalising data based on time.

    This class provides a method to normalise the data based on time.
    It scales the data to the range [0, 1] based on the time.
    """

    def __init__(self, n_frames: int = 100):
        """Initializes a new instance of the LinearTimeNormaliser class.

        Args:
            n_frames: The number of frames to time-normalise the data to.
        """
        self.n_frames: int = n_frames

    def normalise(
        self, trial: model.Trial | model.TrialCycles
    ) -> model.Trial | model.TrialCycles:
        """Normalises the data based on time.

        Args:
            trial: The trial to be normalised.

        Returns:
            model.Trial: A new trial containing the time-normalised data.
            model.TrialCycles: A new segmented trial containing the
            time-normalised data.
        """
        if type(trial) is model.TrialCycles:
            trial = self._normalise_cycle(trial)
        elif type(trial) is model.Trial:
            trial = self._normalise_trial(trial)
        return trial

    def _normalise_trial(self, trial: model.Trial) -> model.Trial:
        """Normalises the data of a Trial based on time.

        Args:
            trial: The trial to be normalised.

        Returns: A new trial containing the time-normalised data.

        """
        new_trial = model.Trial()
        for data_category in trial.get_all_data():
            data = trial.get_data(data_category)
            norm_data = data.meca.time_normalize(n_frames=self.n_frames, norm_time=True)

            new_trial.add_data(data_category, norm_data)
        return new_trial

    def _normalise_cycle(self, trial: model.TrialCycles) -> model.TrialCycles:
        """Normalises the data of a segmented Trial based on time.

        Args:
            trial: The trial to be normalised.


        Returns: A new trial containing the time-normalised data.
        """
        norm_cycles = model.TrialCycles()
        for context, cycles in trial.get_all_cycles().items():
            for cycle_id, cycle in cycles.items():
                new_cycle = self._normalise_trial(cycle)
                norm_cycles.add_cycle(context, cycle_id, new_cycle)

        return norm_cycles
