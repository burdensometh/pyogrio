"""Optional geometry acceptance pipeline for read results.

Supplier data may contain invalid polygons, empty or missing geometries, or
structurally corrupt WKB.  This module provides an opt-in pipeline that runs
after WKB parsing to classify each geometry, repair the repairable ones with
GEOS MakeValid (via shapely), and reject or quarantine the rest while
preserving their original FIDs.

Only valid or explicitly kept ("declared keep") geometries remain in the read
result, attribute columns stay aligned with the repair/rejection status of
each feature, and the layer geometry type in the returned metadata is updated
so that geometries whose type changed during repair (e.g. ``Polygon`` ->
``MultiPolygon``) can still be written out correctly.

The pipeline requires shapely >= 2.0; structural repair
(``repair_method="structure"``) additionally requires shapely >= 2.1 and
GEOS >= 3.10.
"""

import warnings

import numpy as np

from pyogrio._compat import HAS_MAKE_VALID_STRUCTURE, HAS_SHAPELY
from pyogrio.raw import read

# ---------------------------------------------------------------------------
# Status values assigned to each input feature by the acceptance pipeline
# ---------------------------------------------------------------------------

#: Geometry parsed and is topologically valid.
STATUS_VALID = "valid"
#: Geometry was topologically invalid but was successfully repaired.
STATUS_REPAIRED = "repaired"
#: Geometry parsed but is empty (e.g. ``POLYGON EMPTY``).
STATUS_EMPTY = "empty"
#: No geometry was supplied for the feature (NULL / None).
STATUS_MISSING = "missing"
#: WKB bytes were supplied but could not be parsed at all.
STATUS_CORRUPT = "corrupt"
#: Geometry is topologically invalid and could not be repaired.
STATUS_INVALID = "invalid"

#: All status values, in reporting order.
ALL_STATUSES = (
    STATUS_VALID,
    STATUS_REPAIRED,
    STATUS_EMPTY,
    STATUS_MISSING,
    STATUS_CORRUPT,
    STATUS_INVALID,
)

#: Statuses that are always kept in the accepted read result.
_ALWAYS_KEPT = (STATUS_VALID, STATUS_REPAIRED)

# map of shapely geometry type ids to pyogrio geometry type names
_TYPE_ID_NAMES = {
    0: "Point",
    1: "LineString",
    2: "LinearRing",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
}

_SINGLE_TO_MULTI = {
    "Point": "MultiPoint",
    "LineString": "MultiLineString",
    "Polygon": "MultiPolygon",
}


def _require_shapely():
    if not HAS_SHAPELY:
        raise ImportError(
            "shapely >= 2.0 is required to use the geometry acceptance pipeline"
        )


class AcceptanceReport:
    """Per-feature outcome of the geometry acceptance pipeline.

    The report has one row per input feature, in the same order as the input
    read result, so it is row-aligned with the original attribute columns.
    When FIDs were read (``return_fids=True``), they are preserved here so
    repaired and rejected features can be traced back to the supplier data.

    Attributes
    ----------
    status : ndarray of str
        The acceptance status of each input feature; one of ``"valid"``,
        ``"repaired"``, ``"empty"``, ``"missing"``, ``"corrupt"`` or
        ``"invalid"`` (invalid and not repairable).
    kept : ndarray of bool
        Whether the feature was kept in the accepted read result.
    reasons : ndarray of str or None
        Human-readable reason for non-valid features (e.g. the GEOS validity
        reason or WKB parse error); None for valid features.
    fids : ndarray of int or None
        The original FID of each input feature, or None if FIDs were not
        read.  Use ``row_index`` for positional alignment in that case.
    """

    def __init__(self, status, kept, reasons, fids=None):
        self._status = np.asarray(status, dtype=object)
        self._kept = np.asarray(kept, dtype=bool)
        self._reasons = np.asarray(reasons, dtype=object)
        self._fids = None if fids is None else np.asarray(fids)

    @property
    def status(self):
        """Acceptance status of each input feature (ndarray of str)."""
        return self._status

    @property
    def kept(self):
        """Boolean mask of features kept in the accepted result."""
        return self._kept

    @property
    def reasons(self):
        """Reason for each non-valid feature (ndarray of str or None)."""
        return self._reasons

    @property
    def fids(self):
        """Original FID of each input feature, or None if not read."""
        return self._fids

    @property
    def row_index(self):
        """Positional index of each input feature (0-based)."""
        return np.arange(len(self))

    def ids(self):
        """Original FIDs if available, otherwise positional row indices."""
        return self._fids if self._fids is not None else self.row_index

    def __len__(self):
        """Return the number of input features described by the report."""
        return len(self._status)

    @property
    def n_total(self):
        """Total number of input features."""
        return len(self)

    @property
    def n_kept(self):
        """Number of features kept in the accepted result."""
        return int(self._kept.sum())

    @property
    def n_rejected(self):
        """Number of features rejected from the accepted result."""
        return int((~self._kept).sum())

    @property
    def n_repaired(self):
        """Number of features whose geometry was repaired."""
        return int((self._status == STATUS_REPAIRED).sum())

    def counts(self):
        """Return the number of features per status, keyed by status."""
        return {status: int((self._status == status).sum()) for status in ALL_STATUSES}

    def fids_with_status(self, *statuses):
        """Return FIDs of features with the given status(es), in input order.

        Falls back to positional row indices if FIDs were not read.
        """
        mask = np.isin(self._status, list(statuses))
        return self.ids()[mask]

    @property
    def kept_fids(self):
        """FIDs (or row indices) of the features kept in the result."""
        return self.ids()[self._kept]

    @property
    def rejected_fids(self):
        """FIDs (or row indices) of the rejected features."""
        return self.ids()[~self._kept]

    @property
    def repaired_fids(self):
        """FIDs (or row indices) of the features that were repaired."""
        return self.fids_with_status(STATUS_REPAIRED)

    @property
    def adjusted_fids(self):
        """Alias of ``repaired_fids``: features whose geometry was adjusted."""
        return self.repaired_fids

    def to_dict(self):
        """Report as a dict of equal-length arrays (e.g. for a DataFrame).

        Keys: ``"row_index"``, ``"fid"``, ``"status"``, ``"kept"``,
        ``"reason"``.  ``"fid"`` contains None values if FIDs were not read.
        """
        fids = self._fids
        if fids is None:
            fids = np.full(len(self), None, dtype=object)
        return {
            "row_index": self.row_index,
            "fid": fids,
            "status": self._status.copy(),
            "kept": self._kept.copy(),
            "reason": self._reasons.copy(),
        }

    def __repr__(self):
        """Summary with the number of features per status."""
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items() if v)
        return f"<AcceptanceReport n={len(self)} kept={self.n_kept} ({counts})>"


class AcceptanceResult:
    """Result of :func:`accept_geometries`.

    Attributes
    ----------
    geometry : ndarray
        WKB of the accepted geometries (repaired where applicable), aligned
        with ``keep_mask`` True entries.
    keep_mask : ndarray of bool
        Mask over the input geometries indicating which were accepted.
    report : AcceptanceReport
        Per-input-feature acceptance outcome.
    parsed_geometries : ndarray
        The parsed shapely geometries of each input feature, after repair
        where applicable; None where the geometry was missing or corrupt.
    """

    __slots__ = ("geometry", "keep_mask", "parsed_geometries", "report")

    def __init__(self, geometry, keep_mask, report, parsed_geometries):
        self.geometry = geometry
        self.keep_mask = keep_mask
        self.report = report
        self.parsed_geometries = parsed_geometries

    def __repr__(self):
        """Summary with the number of accepted and input geometries."""
        return (
            f"<AcceptanceResult accepted={len(self.geometry)} of {len(self.keep_mask)}>"
        )


class QuarantinedFeatures:
    """Rejected features set aside by the acceptance pipeline.

    Attributes
    ----------
    fids : ndarray of int or None
        Original FIDs of the rejected features (None if FIDs were not read).
    geometry : ndarray
        Original WKB of the rejected features, unmodified (including None
        for missing geometries and unparseable bytes for corrupt ones).
    field_data : tuple of ndarray
        Attribute values of the rejected features, aligned with ``fids`` and
        in the same column order as the read result.
    status : ndarray of str
        Rejection status of each feature.
    reasons : ndarray of str or None
        Rejection reason of each feature.
    """

    __slots__ = ("fids", "field_data", "geometry", "reasons", "status")

    def __init__(self, fids, geometry, field_data, status, reasons):
        self.fids = fids
        self.geometry = geometry
        self.field_data = field_data
        self.status = status
        self.reasons = reasons

    def __len__(self):
        """Return the number of quarantined features."""
        return len(self.geometry)

    def __repr__(self):
        """Summary with the number of quarantined features."""
        return f"<QuarantinedFeatures n={len(self)}>"


class AcceptedRead:
    """Read result after applying the geometry acceptance pipeline.

    Contains only features with valid or explicitly kept geometries; the
    attribute columns in ``field_data`` are filtered with the same mask as
    the geometries and FIDs, so all outputs stay row-aligned.  ``meta`` is a
    copy of the input metadata with ``geometry_type`` updated to reflect the
    accepted (possibly repaired) geometries, so the result can be passed
    directly to :func:`pyogrio.raw.write`.

    Attributes
    ----------
    meta : dict
        Metadata of the accepted result (with updated ``geometry_type``).
    fids : ndarray of int or None
        FIDs of the accepted features (None if FIDs were not read).
    geometry : ndarray
        WKB of the accepted geometries (repaired where applicable).
    field_data : tuple of ndarray
        Attribute values of the accepted features.
    report : AcceptanceReport
        Per-input-feature acceptance outcome (covers rejected features too).
    rejected : QuarantinedFeatures or None
        The rejected features, if ``quarantine=True`` was used, else None.
    """

    __slots__ = ("fids", "field_data", "geometry", "meta", "rejected", "report")

    def __init__(self, meta, fids, geometry, field_data, report, *, rejected=None):
        self.meta = meta
        self.fids = fids
        self.geometry = geometry
        self.field_data = field_data
        self.report = report
        self.rejected = rejected

    def as_tuple(self):
        """Return ``(meta, fids, geometry, field_data)`` like ``raw.read``."""
        return (self.meta, self.fids, self.geometry, self.field_data)

    def __iter__(self):
        """Iterate as ``(meta, fids, geometry, field_data)`` like ``raw.read``."""
        # allows: meta, fids, geometry, field_data = accepted
        return iter(self.as_tuple())

    def __repr__(self):
        """Summary with the number of kept and rejected features."""
        return (
            f"<AcceptedRead kept={len(self.geometry)} "
            f"rejected={self.report.n_rejected}>"
        )


def accept_geometries(
    geometry,
    *,
    fids=None,
    repair=True,
    repair_method="structure",
    keep_empty=False,
    keep_missing=False,
    keep_invalid=False,
    keep_corrupt=False,
):
    """Classify, repair and filter an array of WKB geometries.

    Each geometry is classified as structurally corrupt (unparseable WKB),
    topologically invalid, empty, missing, or valid.  Invalid geometries are
    repaired with GEOS MakeValid (via ``shapely.make_valid``) when ``repair``
    is True; those that cannot be repaired to a non-empty valid geometry are
    rejected.  Empty, missing, invalid and corrupt geometries are rejected
    unless explicitly kept via the corresponding ``keep_*`` flag.

    Parameters
    ----------
    geometry : array-like of bytes or None
        The WKB geometries to validate (as returned by ``pyogrio.raw.read``).
    fids : array-like of int, optional (default: None)
        The FIDs corresponding to the geometries, preserved in the report.
        Must have the same length as ``geometry``.
    repair : bool, optional (default: True)
        Whether to attempt repair of invalid geometries.
    repair_method : str, optional (default: "structure")
        The ``shapely.make_valid`` method to use for repairs.  ``"structure"``
        (requires shapely >= 2.1 and GEOS >= 3.10) preserves the geometry
        dimension and drops collapsed parts, which keeps layer geometry types
        consistent; ``"linework"`` preserves all input vertices but may
        return lower-dimension output.  Falls back to ``"linework"`` with a
        warning if ``"structure"`` is not available.
    keep_empty : bool, optional (default: False)
        Keep empty geometries in the result instead of rejecting them.
    keep_missing : bool, optional (default: False)
        Keep features without geometry (None) in the result instead of
        rejecting them.
    keep_invalid : bool, optional (default: False)
        Keep invalid geometries that could not be repaired in the result
        (their original WKB is passed through unmodified).
    keep_corrupt : bool, optional (default: False)
        Keep corrupt (unparseable) WKB in the result.  Note that such
        geometries cannot be parsed or written downstream.

    Returns
    -------
    AcceptanceResult
        With the accepted WKB ``geometry``, the ``keep_mask`` over the input
        (to filter attribute columns), the per-feature ``report``, and the
        ``parsed_geometries`` (post-repair) of each input feature.

    """
    _require_shapely()
    import shapely

    wkb = np.asarray(geometry, dtype=object)
    n = len(wkb)

    if fids is not None:
        fids = np.asarray(fids)
        if len(fids) != n:
            raise ValueError(
                f"'fids' must have the same length as 'geometry' ({len(fids)} != {n})"
            )

    if repair_method not in ("structure", "linework"):
        raise ValueError(
            f"'repair_method' must be 'structure' or 'linework', got {repair_method!r}"
        )

    if repair and repair_method == "structure" and not HAS_MAKE_VALID_STRUCTURE:
        warnings.warn(
            "repair_method='structure' requires shapely >= 2.1 and GEOS >= 3.10; "
            "falling back to repair_method='linework'",
            stacklevel=2,
        )
        repair_method = "linework"

    status = np.empty(n, dtype=object)
    reasons = np.full(n, None, dtype=object)

    # missing: no WKB supplied; corrupt: WKB supplied but unparseable
    # (shapely.from_wkb with on_invalid="ignore" returns None for both)
    no_input = np.array([g is None for g in wkb], dtype=bool)
    parsed = shapely.from_wkb(wkb, on_invalid="ignore")
    unparsed = shapely.is_missing(parsed)
    missing = no_input
    corrupt = ~no_input & unparsed
    usable = ~unparsed

    if corrupt.any():
        # re-parse the corrupt geometries one by one to capture the error
        for i in np.nonzero(corrupt)[0]:
            try:
                shapely.from_wkb(wkb[i])
            except Exception as exc:
                reasons[i] = f"WKB parse error: {exc}"

    empty = np.zeros(n, dtype=bool)
    if usable.any():
        empty[usable] = shapely.is_empty(parsed[usable])

    is_valid = np.zeros(n, dtype=bool)
    if usable.any():
        is_valid[usable] = shapely.is_valid(parsed[usable])

    invalid = usable & ~empty & ~is_valid
    if invalid.any():
        reasons[invalid] = shapely.is_valid_reason(parsed[invalid])

    # repair invalid geometries; the repaired geometries replace the original
    # WKB in the output, everything else passes through unmodified
    out_wkb = wkb.copy()
    final = parsed.copy()
    repaired = np.zeros(n, dtype=bool)

    if repair and invalid.any():
        idx = np.nonzero(invalid)[0]
        if repair_method == "structure":
            fixed = shapely.make_valid(
                parsed[idx], method="structure", keep_collapsed=False
            )
        else:
            fixed = shapely.make_valid(parsed[idx], method="linework")

        fixed_ok = (
            ~shapely.is_missing(fixed)
            & shapely.is_valid(fixed)
            & ~shapely.is_empty(fixed)
        )
        if fixed_ok.any():
            ok_idx = idx[fixed_ok]
            out_wkb[ok_idx] = shapely.to_wkb(fixed[fixed_ok])
            final[ok_idx] = fixed[fixed_ok]
            repaired[ok_idx] = True
        if (~fixed_ok).any():
            for i in idx[~fixed_ok]:
                reasons[i] = (
                    f"{reasons[i]} | could not be repaired to a non-empty "
                    "valid geometry"
                )

    status[missing] = STATUS_MISSING
    status[corrupt] = STATUS_CORRUPT
    status[empty] = STATUS_EMPTY
    status[invalid] = np.where(
        repaired[invalid], STATUS_REPAIRED, STATUS_INVALID
    ).astype(object)
    status[usable & ~empty & is_valid] = STATUS_VALID

    kept = np.isin(status, list(_ALWAYS_KEPT))
    if keep_empty:
        kept |= status == STATUS_EMPTY
    if keep_missing:
        kept |= status == STATUS_MISSING
    if keep_invalid:
        kept |= status == STATUS_INVALID
    if keep_corrupt:
        kept |= status == STATUS_CORRUPT

    report = AcceptanceReport(status, kept, reasons, fids)
    return AcceptanceResult(out_wkb[kept], kept, report, final)


def _derive_geometry_type(geometries):
    """Derive a pyogrio layer geometry type from shapely geometries.

    Mirrors the inference used when writing: a single type is used as-is,
    mixed single and multi types of the same family become the multi type,
    and anything else (or mixed dimensions) becomes "Unknown".  Returns None
    if there are no geometries to derive from.
    """
    import shapely

    if len(geometries) == 0:
        return None

    names = {
        _TYPE_ID_NAMES[type_id]
        for type_id in np.unique(shapely.get_type_id(geometries))
    }

    if "LinearRing" in names:
        # not supported as an OGR layer geometry type
        return "Unknown"

    if len(names) == 1:
        base = names.pop()
    else:
        families = {name.removeprefix("Multi") for name in names}
        if len(families) == 1:
            family = families.pop()
            if names <= {family, f"Multi{family}"}:
                base = _SINGLE_TO_MULTI[family]
            else:
                base = "Unknown"
        else:
            base = "Unknown"

    if base == "Unknown":
        return base

    has_z = shapely.has_z(geometries)
    if has_z.all():
        return f"{base} Z"
    if has_z.any():
        # mixed 2D and 3D cannot be declared as a layer type
        return "Unknown"
    return base


def accept_read_result(
    result,
    /,
    *,
    quarantine=False,
    repair=True,
    repair_method="structure",
    keep_empty=False,
    keep_missing=False,
    keep_invalid=False,
    keep_corrupt=False,
):
    """Apply the geometry acceptance pipeline to a ``raw.read`` result.

    Parameters
    ----------
    result : tuple of (meta, fids, geometry, field_data)
        A read result as returned by :func:`pyogrio.raw.read`.  Pass
        ``return_fids=True`` when reading to preserve the original FIDs in
        the report and quarantine output.
    quarantine : bool, optional (default: False)
        If True, the rejected features (with their original FIDs, unmodified
        WKB and attribute values) are preserved in the ``rejected`` attribute
        of the result instead of only being recorded in the report.
    repair, repair_method, keep_empty, keep_missing, keep_invalid, keep_corrupt
        See :func:`accept_geometries`.

    Returns
    -------
    AcceptedRead
        Contains only the features with valid or explicitly kept geometries,
        with attribute columns filtered by the same mask, a per-feature
        ``report`` aligned with the input rows, and ``meta["geometry_type"]``
        updated to the (possibly changed) types of the accepted geometries so
        the result can be written out correctly.

    """
    meta, fids, geometry, field_data = result

    if geometry is None:
        raise ValueError(
            "The read result does not contain geometries; the geometry "
            "acceptance pipeline requires read_geometry=True"
        )

    _require_shapely()
    import shapely

    if fids is None:
        warnings.warn(
            "The read result does not include FIDs; the report will use "
            "positional row indices instead of the original FIDs. Read with "
            "return_fids=True to preserve the original FIDs.",
            stacklevel=2,
        )

    accepted = accept_geometries(
        geometry,
        fids=fids,
        repair=repair,
        repair_method=repair_method,
        keep_empty=keep_empty,
        keep_missing=keep_missing,
        keep_invalid=keep_invalid,
        keep_corrupt=keep_corrupt,
    )
    keep = accepted.keep_mask

    new_meta = dict(meta)
    parseable = keep & ~shapely.is_missing(accepted.parsed_geometries)
    derived = _derive_geometry_type(accepted.parsed_geometries[parseable])
    if derived is not None:
        new_meta["geometry_type"] = derived

    if field_data is not None:
        new_field_data = tuple(np.asarray(column)[keep] for column in field_data)
    else:
        new_field_data = field_data

    new_fids = None if fids is None else np.asarray(fids)[keep]

    rejected = None
    if quarantine:
        drop = ~keep
        rejected = QuarantinedFeatures(
            fids=None if fids is None else np.asarray(fids)[drop],
            geometry=np.asarray(geometry, dtype=object)[drop],
            field_data=(
                tuple(np.asarray(column)[drop] for column in field_data)
                if field_data is not None
                else ()
            ),
            status=accepted.report.status[drop],
            reasons=accepted.report.reasons[drop],
        )

    return AcceptedRead(
        meta=new_meta,
        fids=new_fids,
        geometry=accepted.geometry,
        field_data=new_field_data,
        report=accepted.report,
        rejected=rejected,
    )


def read_accepted(
    path_or_buffer,
    /,
    *,
    quarantine=False,
    repair=True,
    repair_method="structure",
    keep_empty=False,
    keep_missing=False,
    keep_invalid=False,
    keep_corrupt=False,
    **kwargs,
):
    """Read an OGR data source and apply the geometry acceptance pipeline.

    This reads the data source with :func:`pyogrio.raw.read` (all of its
    parameters are supported and passed through) and then classifies each
    geometry as valid, repaired, empty, missing, corrupt or invalid,
    repairing invalid geometries where possible.  Only valid or explicitly
    kept geometries remain in the result; rejected features are recorded in
    the report (and optionally quarantined) with their original FIDs.

    FIDs are always read (``return_fids=True``) so that the original FID of
    every repaired or rejected feature is preserved.

    Parameters
    ----------
    path_or_buffer : pathlib.Path or str, or bytes buffer
        A dataset path or URI, raw buffer, or file-like object with a read
        method.
    quarantine : bool, optional (default: False)
        If True, preserve the rejected features (original FIDs, unmodified
        WKB and attribute values) in the ``rejected`` attribute of the
        result.
    repair, repair_method, keep_empty, keep_missing, keep_invalid, keep_corrupt
        See :func:`accept_geometries`.
    **kwargs
        Additional parameters are passed to :func:`pyogrio.raw.read`
        (e.g. ``layer``, ``where``, ``columns``, ``skip_features``).

    Returns
    -------
    AcceptedRead
        The accepted read result.  ``result.meta``, ``result.fids``,
        ``result.geometry`` and ``result.field_data`` mirror the output of
        :func:`pyogrio.raw.read` (and can be unpacked the same way), so the
        result can be passed straight to :func:`pyogrio.raw.write`;
        ``result.report`` describes the acceptance outcome of every input
        feature, aligned with the original attribute columns.

    Examples
    --------
    >>> import pyogrio
    >>> result = pyogrio.read_accepted("supplier_data.gpkg", quarantine=True)
    >>> result.report.counts()  # doctest: +SKIP
    {'valid': 95, 'repaired': 3, 'empty': 1, 'missing': 0, 'corrupt': 0,
     'invalid': 1}
    >>> result.report.repaired_fids  # which geometries were adjusted  # doctest: +SKIP
    array([12, 37, 88])
    >>> pyogrio.raw.write("cleaned.gpkg", result.geometry, result.field_data,
    ...                   **result.meta)  # doctest: +SKIP

    """
    if kwargs.get("return_fids") is False:
        warnings.warn(
            "read_accepted always reads FIDs to preserve them in the report; "
            "ignoring return_fids=False",
            stacklevel=2,
        )
    kwargs.pop("return_fids", None)

    result = read(path_or_buffer, return_fids=True, **kwargs)

    return accept_read_result(
        result,
        quarantine=quarantine,
        repair=repair,
        repair_method=repair_method,
        keep_empty=keep_empty,
        keep_missing=keep_missing,
        keep_invalid=keep_invalid,
        keep_corrupt=keep_corrupt,
    )
