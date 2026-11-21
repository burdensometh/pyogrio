"""Support for the optional dual-column (UTC + original offset) datetime contract.

GDAL stores a date-time as a *local wall clock* time plus a UTC offset (the
``TZFlag`` of an :c:type:`OGRField`, in multiples of 15 minutes).  Once such a
column is unified to UTC, the original local time can no longer be recovered,
and a column that mixes time zones - or that crosses a daylight-saving change -
cannot be represented by a single ``datetime64`` column at all.

The helpers in this module implement an optional contract that keeps both
pieces instead of choosing one.  For a date-time column ``<name>`` the data is
returned as two columns:

``<name>``
    the exact instant in UTC.  numpy: ``datetime64[ms]`` (the values are UTC),
    Arrow: ``timestamp[ms, tz=UTC]``, pandas: ``datetime64[ms, UTC]``.
``<name>_offset``
    the UTC offset that was stored in the file, as a duration
    (``timedelta64[ms]`` / ``duration[ms]``).  ``NaT`` / ``null`` means the file
    did **not** record a time zone (unknown), while ``0`` means it recorded an
    explicit zero offset (UTC).  The two cases are deliberately kept apart.

Values are carried and converted per row, next to their own offset, so that:

- batching can never shift an offset onto another row;
- the two ``01:30`` wall clock readings of a DST fall-back stay two distinct
  instants (different UTC values and offsets);
- ``null`` values stay ``null`` in both columns;
- date-times or offsets whose precision exceeds what GDAL can store
  (sub-millisecond date-times, offsets that are not a whole multiple of 15
  minutes or outside the supported range) are rejected explicitly instead of
  being silently rounded.
"""

import datetime as _dt
import re

import numpy as np

#: Value of the ``datetime_layout`` keyword that selects the contract.
DATETIME_LAYOUT_UTC_OFFSET = "utc_offset"

#: Suffix of the offset column that accompanies a date-time column.
DATETIME_OFFSET_SUFFIX = "_offset"

VALID_DATETIME_LAYOUTS = (DATETIME_LAYOUT_UTC_OFFSET,)

#: Placeholder OGR type reported in ``meta`` for offset columns.  It is not a
#: GDAL type; it only marks the companion of a date-time column.
OFT_DATETIME_OFFSET = "OFTDateTimeOffset"

# GDAL's TZFlag encodes whole quarters of an hour, with 100 meaning GMT.
# https://gdal.org/development/rfc/rfc56_millisecond_precision.html#core-changes
GDAL_TZ_GMT = 100
GDAL_TZ_QUARTER_MINUTES = 15

#: Offsets outside this range cannot be expressed by GDAL's TZFlag.
MIN_OFFSET_MINUTES = -12 * 60
MAX_OFFSET_MINUTES = 14 * 60

_NANOS = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 60 * 60 * 1_000_000_000,
    "D": 24 * 60 * 60 * 1_000_000_000,
}
_NANOS_PER_MS = _NANOS["ms"]
_NAT_INT = np.iinfo("int64").min

_TZ_SUFFIX_RE = re.compile(r"(?P<tz>Z|[+-][0-9]{2}(?::?[0-9]{2})?)\s*$", re.IGNORECASE)


def validate_layout(datetime_layout, datetime_as_string=False):
    """Validate a ``datetime_layout`` keyword value."""
    if datetime_layout is None:
        return None
    if datetime_layout not in VALID_DATETIME_LAYOUTS:
        raise ValueError(
            f"'datetime_layout' must be one of {list(VALID_DATETIME_LAYOUTS)} or "
            f"None, got {datetime_layout!r}"
        )
    if datetime_as_string:
        raise ValueError(
            "'datetime_layout' and 'datetime_as_string' cannot be used together"
        )
    return datetime_layout


def is_enabled(datetime_layout):
    return datetime_layout == DATETIME_LAYOUT_UTC_OFFSET


def offset_column_name(name):
    """Return the offset column name that accompanies date-time column ``name``."""
    return f"{name}{DATETIME_OFFSET_SUFFIX}"


def base_column_name(name):
    """Return the date-time column name for offset column ``name``."""
    return name[: -len(DATETIME_OFFSET_SUFFIX)]


def is_offset_column_name(name):
    """Return whether ``name`` ends with the offset suffix."""
    return isinstance(name, str) and name.endswith(DATETIME_OFFSET_SUFFIX)


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    # pandas.NaT / pandas.NA
    return type(value).__name__ in ("NaTType", "NAType")


def _parse_offset(tz):
    """Convert ``Z`` / ``+05:45`` / ``-0400`` to minutes; ``None`` if absent."""
    if tz is None:
        return None
    if tz.upper() == "Z":
        return 0
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    hours = int(digits[:2])
    minutes = int(digits[2:4]) if len(digits) > 2 else 0
    if minutes >= 60:
        raise ValueError(f"invalid time zone offset {tz!r}")
    return sign * (hours * 60 + minutes)


def split_iso_datetime(value):
    """Split an ISO8601 date-time string into a naive datetime and its offset.

    Parameters
    ----------
    value : str
        Date-time string as emitted by GDAL, e.g. ``"2024/11/03 01:30:00-04"``
        (GDAL < 3.7), ``"2024-01-01T00:00:00Z"`` or ``"2024-01-01T00:00:00"``.

    Returns
    -------
    (datetime.datetime, int or None)
        The wall clock time without time zone information, and the UTC offset in
        minutes.  ``None`` means the string carries no offset at all.
    """
    text = value.strip()
    # GDAL before 3.7 uses "YYYY/MM/DD HH:MM:SS+HH" instead of ISO8601.
    if len(text) > 10 and text[4] == "/":
        text = text.replace("/", "-", 2)
    if len(text) > 10 and text[10] == " ":
        text = f"{text[:10]}T{text[11:]}"

    match = _TZ_SUFFIX_RE.search(text)
    tz = None
    if match is not None:
        tz = match.group("tz")
        text = text[: match.start()].rstrip()
    elif text.endswith("Z"):
        tz = "Z"
        text = text[:-1].rstrip()

    try:
        wall = _dt.datetime.fromisoformat(text)
    except ValueError as ex:
        raise ValueError(f"could not parse date-time value {value!r}: {ex}") from None

    return wall.replace(tzinfo=None), _parse_offset(tz)


def parse_datetime_array(values, name=None):
    """Parse date-time strings into UTC instants and original offsets.

    Every value is parsed on its own, so an offset can never leak from one row
    into the next regardless of how the data is batched.

    Parameters
    ----------
    values : sequence
        Date-time strings; ``None``, ``NaN``, ``NaT`` and ``pd.NA`` are missing.
    name : str, optional
        Column name, only used in error messages.

    Returns
    -------
    (utc, offsets)
        ``utc`` is ``datetime64[ms]`` with the exact instants.  ``offsets`` is
        ``timedelta64[ms]`` with the original offset, ``NaT`` where the file did
        not record one.  Rows without an offset are treated as already being in
        UTC (the same assumption as ``mixed_offsets_as_utc=True``).
    """
    count = len(values)
    utc = np.full(count, np.datetime64("NaT"), dtype="datetime64[ms]")
    offsets = np.full(count, np.timedelta64("NaT"), dtype="timedelta64[ms]")

    for i in range(count):
        value = values[i]
        if _is_missing(value):
            continue
        if not isinstance(value, str):
            where = f" in column {name!r}" if name is not None else ""
            raise TypeError(
                f"expected date-time strings to parse{where}, got "
                f"{type(value).__name__}"
            )
        try:
            wall, offset = split_iso_datetime(value)
        except ValueError as ex:
            where = f" in column {name!r}" if name is not None else ""
            raise ValueError(f"{ex}{where}") from None

        wall64 = np.datetime64(wall, "ms")
        utc[i] = wall64
        if offset is not None:
            offsets[i] = np.timedelta64(offset, "m")
            utc[i] = wall64 - np.timedelta64(offset, "m")

    return utc, offsets


def format_offset(offset_minutes):
    """Format an offset in minutes as ``+HH:MM`` / ``-HH:MM``."""
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(int(offset_minutes)), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _unit_nanos(dtype):
    """Return the nanoseconds per array tick, or None for a non-datetime dtype."""
    if np.dtype(dtype).kind not in ("M", "m"):
        return None
    unit = np.datetime_data(np.dtype(dtype))[0]
    return _NANOS.get(unit)


def _check_datetime_precision(values, name):
    nanos = _unit_nanos(values.dtype)
    if nanos is None:
        raise TypeError(
            f"expected datetime64 data for column {name!r}, got {values.dtype!r}"
        )
    # A tick of at least a millisecond cannot express sub-millisecond precision,
    # so only finer units need checking for values a millisecond cannot represent.
    if nanos >= _NANOS_PER_MS or values.size == 0:
        return
    ticks = values.astype("int64")
    valid = ticks != _NAT_INT
    # ticks per millisecond: nanosecond ticks need 1_000_000 of them per ms
    ticks_per_ms = _NANOS_PER_MS // nanos
    if np.any((ticks[valid] % ticks_per_ms) != 0):
        raise ValueError(
            f"column {name!r} contains date-times that are more precise than "
            f"milliseconds, which GDAL cannot store without truncating them; "
            f"round the values to milliseconds first"
        )


def _check_offsets(values, name):
    nanos = _unit_nanos(values.dtype)
    if nanos is None:
        raise TypeError(
            f"expected timedelta64 data for column {name!r}, got {values.dtype!r}"
        )
    valid = ~np.isnat(values)
    if not np.any(valid):
        return

    if nanos < _NANOS_PER_MS:
        ticks = values.astype("int64")
        ticks_per_ms = _NANOS_PER_MS // nanos
        if np.any((ticks[valid] % ticks_per_ms) != 0):
            raise ValueError(
                f"column {name!r} contains time zone offsets that are more "
                f"precise than milliseconds; GDAL cannot store them"
            )

    # TZFlag can only encode whole minutes, in multiples of 15.
    minutes = values.astype("timedelta64[m]")
    as_minutes = minutes.astype("timedelta64[ms]")
    not_whole = values[valid].astype("timedelta64[ms]") != as_minutes[valid]
    if np.any(not_whole):
        bad = values[valid][not_whole][0]
        raise ValueError(
            f"column {name!r} contains the time zone offset {bad}, which is not "
            f"a whole number of minutes; GDAL cannot store it"
        )

    integer_minutes = minutes.astype("int64")
    selected = integer_minutes[valid]
    not_quarter = (selected % GDAL_TZ_QUARTER_MINUTES) != 0
    if np.any(not_quarter):
        bad = int(selected[not_quarter][0])
        raise ValueError(
            f"column {name!r} contains a time zone offset of {bad} minutes, "
            f"which is not a multiple of 15 minutes; GDAL cannot store it "
            f"without losing information"
        )
    outside = (selected < MIN_OFFSET_MINUTES) | (selected > MAX_OFFSET_MINUTES)
    if np.any(outside):
        bad = int(selected[outside][0])
        raise ValueError(
            f"column {name!r} contains a time zone offset of {bad} minutes, "
            f"which is outside the range GDAL can store "
            f"({MIN_OFFSET_MINUTES} to {MAX_OFFSET_MINUTES} minutes)"
        )


def utc_to_wall_and_flags(utc_values, offset_values, name):
    """Combine UTC instants and original offsets into wall clock values.

    Parameters
    ----------
    utc_values : array-like of datetime64
        The exact instants, in UTC.
    offset_values : array-like of timedelta64
        The original UTC offsets; ``NaT`` means unknown.
    name : str
        Column name used in error messages.

    Returns
    -------
    (wall, flags)
        ``wall`` is ``datetime64[ms]`` with the local wall clock times (``NaT``
        where the instant is missing).  ``flags`` is ``float64`` with the GDAL
        TZFlag per row (``NaN`` for an unknown offset).

    Raises
    ------
    ValueError
        If the values cannot be stored by GDAL exactly.
    """
    utc = np.asarray(utc_values)
    offsets = np.asarray(offset_values)
    if len(utc) != len(offsets):
        raise ValueError(
            f"date-time column {name!r} and its offset column have different "
            f"lengths ({len(utc)} != {len(offsets)})"
        )

    _check_datetime_precision(utc, name)
    _check_offsets(offsets, name)

    utc = utc.astype("datetime64[ms]")
    offsets = offsets.astype("timedelta64[ms]")

    known = ~np.isnat(offsets)
    # Where the offset is unknown the stored wall clock time is the UTC instant
    # itself (GDAL keeps it as a naive time with TZFlag 0), so it round-trips.
    effective_offsets = np.where(known, offsets.astype("int64"), 0).astype(
        "timedelta64[ms]"
    )
    wall = utc + effective_offsets
    wall[np.isnat(utc)] = np.datetime64("NaT")

    minutes = offsets.astype("timedelta64[m]").astype("int64")
    flags = np.full(len(utc), np.nan, dtype="float64")
    flags[known] = GDAL_TZ_GMT + minutes[known] / GDAL_TZ_QUARTER_MINUTES
    return wall, flags


def format_utc_offset_strings(utc_values, offset_values):
    """Format UTC instants and original offsets as ISO8601 wall clock strings.

    Used for the Arrow write path: GDAL's Arrow writer cannot attach a different
    offset to each row of a timestamp column, but it does preserve the offset of
    a date-time *string*.  Each value is rendered as its local wall clock time
    (``utc + offset``) with an explicit ``+HH:MM`` suffix.  A missing offset
    renders as a naive string, which GDAL reads back as an unknown time zone.
    """
    utc = np.asarray(utc_values).astype("datetime64[ms]")
    offsets = np.asarray(offset_values).astype("timedelta64[ms]")
    minutes = offsets.astype("timedelta64[m]").astype("int64")
    known = ~np.isnat(offsets)
    # unknown offsets are rendered as the naive UTC instant (GDAL reads those
    # back as an unknown time zone), not as a missing value
    effective_offsets = np.where(known, offsets.astype("int64"), 0).astype(
        "timedelta64[ms]"
    )
    wall = utc + effective_offsets
    wall[np.isnat(utc)] = np.datetime64("NaT")

    result = []
    for i in range(len(utc)):
        if np.isnat(utc[i]):
            result.append(None)
            continue
        text = str(wall[i]).replace(" ", "T")
        if known[i]:
            text += format_offset(minutes[i])
        result.append(text)
    return result


def driver_creation_field_types(driver):
    """Return the ``DMD_CREATIONFIELDDATATYPES`` types of ``driver``, or None."""
    from pyogrio._ogr import _get_driver_metadata_item

    if driver is None:
        return None
    try:
        types = _get_driver_metadata_item(driver, "DMD_CREATIONFIELDDATATYPES")
    except Exception:
        return None
    if not types:
        return None
    return types.split()


def validate_driver_supports_datetime_offset(driver):
    """Reject drivers that cannot store a date-time with a time zone offset."""
    types = driver_creation_field_types(driver)
    if types is not None and "DateTime" in types:
        return

    from pyogrio.errors import DataSourceError

    detail = f" (supported types: {' '.join(types)})" if types else ""
    raise DataSourceError(
        f"driver {driver!r} cannot store date-times with a time zone offset"
        f"{detail}. Use a driver that supports the DateTime field type (for "
        f"example GPKG, GeoJSON or FlatGeobuf), or write the values as strings."
    )


def find_offset_columns(fields, dtypes_by_name=None):
    """Map each date-time column with an ``_offset`` companion to that companion.

    Parameters
    ----------
    fields : sequence of str
        Field names in the same order as the field data.
    dtypes_by_name : dict, optional
        Numpy/pandas dtypes keyed by field name. When provided, a companion is
        only accepted if the date-time column is a datetime dtype and the
        companion is a timedelta dtype, so an unrelated attribute that merely
        happens to end with ``_offset`` is not mistaken for a companion.

    Returns
    -------
    dict
        ``{datetime_name: offset_name}`` for every column that has an
        ``_offset`` companion in ``fields``.
    """
    field_set = set(fields)
    mapping = {}
    for name in fields:
        if not isinstance(name, str) or name.endswith(DATETIME_OFFSET_SUFFIX):
            continue
        companion = offset_column_name(name)
        if companion not in field_set:
            continue
        if dtypes_by_name is not None:
            if not _is_datetime_dtype(dtypes_by_name.get(name)):
                continue
            if not _is_timedelta_dtype(dtypes_by_name.get(companion)):
                continue
        mapping[name] = companion
    return mapping


def _dtype_kind(dtype):
    if dtype is None:
        return None
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind
    return np.dtype(dtype).kind


def _is_datetime_dtype(dtype):
    return _dtype_kind(dtype) == "M"


def _is_timedelta_dtype(dtype):
    return _dtype_kind(dtype) == "m"


def prepare_write(utc_values, offset_values, name):
    """Convert a UTC column and its offset column for writing.

    Parameters
    ----------
    utc_values : ndarray of datetime64
        The exact instants, in UTC.
    offset_values : ndarray of timedelta64
        The original UTC offsets; ``NaT`` means unknown.
    name : str
        Column name used in error messages.

    Returns
    -------
    (wall, gdal_flags)
        ``wall`` is ``datetime64[ms]`` with the local wall clock times, and
        ``gdal_flags`` is a float64 array with the GDAL TZFlag for each row
        (``NaN`` for an unknown offset).

    Raises
    ------
    ValueError
        If the values cannot be stored by GDAL exactly.
    """
    return utc_to_wall_and_flags(utc_values, offset_values, name)
