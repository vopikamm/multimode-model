"""Implementation of border API."""
import numpy as np
from typing import Sequence, Type

from .api import (
    RegularSplitMergerBase,
    BorderBase,
    BorderMergerBase,
    BorderSplitterBase,
    TailorBase,
)
from .datastructure import (
    Domain,
    State,
    Parameter,
)

# from redis import Redis


class RegularSplitMerger(RegularSplitMergerBase[np.ndarray]):
    """Implements splitting and merging into regular grid."""

    def split_array(self, array: np.ndarray) -> tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        tuple[np.ndarray, ...]
        """
        return np.array_split(array, indices_or_sections=self.parts, axis=self.dim[0])

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        return np.concatenate(arrays, axis=self.dim[0])


class BorderSplitter(BorderSplitterBase[np.ndarray]):
    """Implements splitting off stripes along a dimension."""

    def split_array(self, array: np.ndarray) -> tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        tuple[np.ndarray, ...]
        """
        slices = array.ndim * [slice(None)]
        slices[self._axis] = self._slice
        return (array[tuple(slices)],)


class Border(BorderBase[State, Parameter], Domain):
    """Implementation of Border class from API on State class."""

    @staticmethod
    def _provide_border_splitter():
        return BorderSplitter


class BorderMerger(BorderMergerBase[np.ndarray, Border]):
    """Merges the borders with a Domain along a dimension.

    This merger is suppose to be used in the merge classmethod of the
    DomainState class. The order of arguments must be
    (left_border, domain, right_border).
    """

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        slices_center = arrays[1].ndim * [slice(None)]
        slices_center[self._axis] = self._slice_center

        left, base, right = arrays
        out = np.concatenate((left, base[tuple(slices_center)], right), axis=self._axis)
        return out


class Tail(TailorBase[Domain, Border]):
    """Implement Tailor class from API."""

    @staticmethod
    def _provide_border_type() -> Type[Border]:
        return Border

    @staticmethod
    def stitch(
        base: Domain,
        borders: tuple[Border, Border],
    ) -> Domain:
        """Implement stitch method from API.

        borders need to be ordered left_border, right_border
        """
        left_border, right_border = borders
        border_merger = BorderMerger.from_borders(left_border, right_border)

        if base.iteration == left_border.iteration == right_border.iteration:
            assert base.id == left_border.id == right_border.id
        else:
            raise ValueError(
                "Borders iteration mismatch. Left: {}, right: {}, domain: {}".format(
                    left_border.iteration,
                    right_border.iteration,
                    base.iteration,
                )
            )
        # necessary for caching of split / merge operations since
        # hashing of ParameterSplit and GridSplit is id based.
        vars = {
            v: base._state_type()._variable_type()(
                data=border_merger.merge_array(
                    tuple(
                        getattr(o.state, v).safe_data
                        for o in (left_border, base, right_border)
                    )
                ),
                grid=base.state.__getattribute__(v).grid,
                time=base.state.__getattribute__(v).time,
            )
            for v in base.state.variables
        }
        merged_state = State(**vars)
        return Domain(
            state=merged_state,
            history=base.history,
            parameter=base.parameter,
            iteration=base.iteration,
            id=base.id,
        )


# def _dump_to_redis(domain: DomainState):
#     r = Redis(host="localhost", port="6379", db="0")

#     if r.ping():
#         flag = int(r.get("_avg_eta"))

#         if flag == 1:
#             k = format(domain.id, "05d") + "_" + format(domain.it, "05d") + "_eta"
#             h, w = domain.eta.safe_data.shape
#             shape = pack(">II", h, w)
#             encoded = shape + domain.eta.safe_data.tobytes()

#             r.set(k, encoded)
