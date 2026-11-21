"""Low level functions to read and write OGR data sources."""

import warnings
from io import BytesIO
from pathlib import Path

import numpy as np

from pyogrio import _datetime
from pyogrio._compat import HAS_ARROW_WRITE_API, HAS_PYARROW
from pyogrio._env import GDALEnv
from pyogrio.core import detect_write_driver
from pyogrio.errors import DataSourceError
from pyogrio.util import (
    _mask_to_wkb,
    _preprocess_options_key_value,
    get_vsi_path_or_buffer,
    vsi_path,
)

with GDALEnv():
    from pyogrio._io import ogr_open_arrow, ogr_read, ogr_write, ogr_write_arrow
    from pyogrio._ogr import (
        _get_driver_metadata_item,
        get_gdal_version,
        get_gdal_version_string,
        ogr_driver_supports_vsi,
        ogr_driver_supports_write,
    )


DRIVERS_NO_MIXED_SINGLE_MULTI = {
    "FlatGeobuf",
    "GPKG",
}

DRIVERS_NO_MIXED_DIMENSIONS = {
    "FlatGeobuf",
}


def _add_utc_offset_columns(meta, field_data):
    """Replace DateTime fields by a UTC column plus an offset column.

    Each DateTime field is read as an ISO8601 string and paired with a
    ``<name>_offset`` column holding the original UTC offset, so the local time
    can be recovered without losing the exact instant.  The offset is missing
    (``NaT``) when the file did not record a time zone and zero when it recorded
    an explicit UTC offset; the two are deliberately kept apart.

    Date (``OFTDate``) fields, which are also returned as strings when reading
    DateTime fields as strings, are converted back to ``datetime64[D]``.

    Parameters
    ----------
    meta : dict
        Read metadata.
    field_data : tuple of ndarrays
        The field data as returned by ``ogr_read`` with ``datetime_as_string``.

    Returns
    -------
    (dict, tuple of ndarrays)
        The updated metadata and field data, with each DateTime field replaced by
        a UTC datetime64 column followed by its offset column.
    """
    names = list(meta["fields"])
    dtypes = list(meta["dtypes"])
    ogr_types = list(meta["ogr_types"])
    ogr_subtypes = list(meta["ogr_subtypes"])

    new_names = []
    new_dtypes = []
    new_ogr_types = []
    new_ogr_subtypes = []
    new_field_data = []

    for name, dtype, ogr_type, ogr_subtype, values in zip(
        names, dtypes, ogr_types, ogr_subtypes, field_data, strict=True
    ):
        if ogr_type == "OFTDateTime":
            utc, offsets = _datetime.parse_datetime_array(values, name)
            new_names.append(name)
            new_dtypes.append("datetime64[ms]")
            new_ogr_types.append(ogr_type)
            new_ogr_subtypes.append(ogr_subtype)
            new_field_data.append(utc)

            new_names.append(_datetime.offset_column_name(name))
            new_dtypes.append("timedelta64[ms]")
            new_ogr_types.append(_datetime.OFT_DATETIME_OFFSET)
            new_ogr_subtypes.append("OFSTNone")
            new_field_data.append(offsets)
            continue

        if ogr_type == "OFTDate":
            # datetime_as_string also stringifies Date fields; restore the dtype
            values = np.array(
                [None if v is None else np.datetime64(v, "D") for v in values],
                dtype="datetime64[D]",
            )
            dtype = "datetime64[D]"

        new_names.append(name)
        new_dtypes.append(dtype)
        new_ogr_types.append(ogr_type)
        new_ogr_subtypes.append(ogr_subtype)
        new_field_data.append(values)

    meta = dict(meta)
    meta["fields"] = np.asarray(new_names, dtype=object)
    meta["dtypes"] = np.asarray(new_dtypes, dtype=object)
    meta["ogr_types"] = new_ogr_types
    meta["ogr_subtypes"] = new_ogr_subtypes
    meta["datetime_layout"] = _datetime.DATETIME_LAYOUT_UTC_OFFSET
    return meta, tuple(new_field_data)


def _datetime_field_names(meta):
    return [
        name
        for name, ogr_type in zip(meta["fields"], meta["ogr_types"], strict=True)
        if ogr_type == "OFTDateTime"
    ]


def _read_datetime_strings_by_fid(path_or_buffer, names, read_kwargs):
    """Read DateTime fields as strings, keyed by FID.

    The Arrow API cannot carry a per-row UTC offset for a timestamp column on
    every GDAL version, so the DateTime fields are read a second time through the
    string-based reader.  The result is keyed by FID, so it stays aligned with
    the Arrow rows even when the two APIs return the features in a different
    order (for example when ``fids`` is used).
    """
    read_kwargs = dict(read_kwargs)
    # the DateTime values are re-read as strings for exactly these columns
    read_kwargs["columns"] = names
    _, fids, _, field_data = read(
        path_or_buffer,
        read_geometry=False,
        return_fids=True,
        datetime_as_string=True,
        **read_kwargs,
    )
    by_fid = {}
    for position, fid in enumerate(fids):
        by_fid[int(fid)] = {
            name: values[position]
            for name, values in zip(names, field_data, strict=True)
        }
    return by_fid


def _align_by_fid(meta, table, by_fid):
    """Return a list of the rows of ``by_fid`` in the row order of ``table``."""
    fid_column = meta.get("fid_column") or "OGC_FID"
    if fid_column not in table.column_names:
        raise RuntimeError(
            "the FID column is missing from the Arrow data, so the DateTime "
            "offsets cannot be aligned; use the non-Arrow API instead"
        )
    fids = table.column(fid_column).to_pylist()
    try:
        return [by_fid[int(fid)] for fid in fids]
    except KeyError:
        raise RuntimeError(
            "could not align the Arrow rows with the DateTime offsets read "
            "through the non-Arrow API (the FID values differ between the two "
            "reads); use the non-Arrow API instead"
        ) from None


def _convert_table_datetime_layout(meta, table, by_fid):
    """Replace DateTime columns of an Arrow table with UTC + offset columns."""
    import pyarrow as pa

    aligned = _align_by_fid(meta, table, by_fid)

    replacements = {}
    for name in _datetime_field_names(meta):
        values = [row[name] for row in aligned]
        utc, offsets = _datetime.parse_datetime_array(values, name)
        offset_name = _datetime.offset_column_name(name)
        replacements[name] = [
            (name, pa.array(utc).cast(pa.timestamp("ms", tz="UTC"))),
            (offset_name, pa.array(offsets)),
        ]

    output = []
    for name in table.column_names:
        entries = replacements.get(name)
        if entries is not None:
            output.extend(entries)
        else:
            output.append((name, table.column(name)))

    return pa.table(dict(output))


def _new_meta_for_datetime_layout(meta):
    """Return metadata describing the UTC + offset columns."""
    names = list(meta["fields"])
    dtypes = list(meta["dtypes"])
    ogr_types = list(meta["ogr_types"])
    ogr_subtypes = list(meta["ogr_subtypes"])

    new_names = []
    new_dtypes = []
    new_ogr_types = []
    new_ogr_subtypes = []
    for name, dtype, ogr_type, ogr_subtype in zip(
        names, dtypes, ogr_types, ogr_subtypes, strict=True
    ):
        new_names.append(name)
        new_dtypes.append(dtype)
        new_ogr_types.append(ogr_type)
        new_ogr_subtypes.append(ogr_subtype)
        if ogr_type == "OFTDateTime":
            new_dtypes[-1] = "datetime64[ms]"
            new_names.append(_datetime.offset_column_name(name))
            new_dtypes.append("timedelta64[ms]")
            new_ogr_types.append(_datetime.OFT_DATETIME_OFFSET)
            new_ogr_subtypes.append("OFSTNone")

    new_meta = dict(meta)
    new_meta["fields"] = np.asarray(new_names, dtype=object)
    new_meta["dtypes"] = np.asarray(new_dtypes, dtype=object)
    new_meta["ogr_types"] = new_ogr_types
    new_meta["ogr_subtypes"] = new_ogr_subtypes
    new_meta["datetime_layout"] = _datetime.DATETIME_LAYOUT_UTC_OFFSET
    return new_meta


def _apply_datetime_layout_to_arrow(path_or_buffer, meta, table, read_kwargs):
    """Apply the dual-column contract to an Arrow table.

    The DateTime values are re-read as strings through the (non-Arrow) reader and
    matched to the Arrow rows by FID, so the offsets stay attached to the right
    feature regardless of the order in which the two APIs return them.
    """
    names = _datetime_field_names(meta)
    if not names:
        return _new_meta_for_datetime_layout(meta), table

    by_fid = _read_datetime_strings_by_fid(path_or_buffer, names, read_kwargs)
    table = _convert_table_datetime_layout(meta, table, by_fid)
    return _new_meta_for_datetime_layout(meta), table


def read(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    force_2d=False,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    datetime_as_string=False,
    datetime_layout=None,
    **kwargs,
):
    """Read OGR data source into numpy arrays.

    IMPORTANT: non-linear geometry types (e.g., MultiSurface) are converted
    to their linear approximations.

    Parameters
    ----------
    path_or_buffer : pathlib.Path or str, or bytes buffer
        A dataset path or URI, raw buffer, or file-like object with a read method.
    layer : int or str, optional (default: first layer)
        If an integer is provided, it corresponds to the index of the layer
        with the data source.  If a string is provided, it must match the name
        of the layer in the data source.  Defaults to first layer in data source.
    encoding : str, optional (default: None)
        If present, will be used as the encoding for reading string values from
        the data source.  By default will automatically try to detect the native
        encoding and decode to ``UTF-8``.
    columns : list-like, optional (default: all columns)
        List of column names to import from the data source.  Column names must
        exactly match the names in the data source, and will be returned in
        the order they occur in the data source.  To avoid reading any columns,
        pass an empty list-like.  If combined with ``where`` parameter, must
        include columns referenced in the ``where`` expression or the data may
        not be correctly read; the data source may return empty results or
        raise an exception (behavior varies by driver).
    read_geometry : bool, optional (default: True)
        If True, will read geometry into WKB.  If False, geometry will be None.
    force_2d : bool, optional (default: False)
        If the geometry has Z values, setting this to True will cause those to
        be ignored and 2D geometries to be returned
    skip_features : int, optional (default: 0)
        Number of features to skip from the beginning of the file before
        returning features.  If greater than available number of features, an
        empty DataFrame will be returned.  Using this parameter may incur
        significant overhead if the driver does not support the capability to
        randomly seek to a specific feature, because it will need to iterate
        over all prior features.
    max_features : int, optional (default: None)
        Number of features to read from the file.
    where : str, optional (default: None)
        Where clause to filter features in layer by attribute values. If the data source
        natively supports SQL, its specific SQL dialect should be used (eg. SQLite and
        GeoPackage: `SQLITE`_, PostgreSQL). If it doesn't, the `OGRSQL WHERE`_ syntax
        should be used. Note that it is not possible to overrule the SQL dialect, this
        is only possible when you use the SQL parameter.
        Examples: ``"ISO_A3 = 'CAN'"``, ``"POP_EST > 10000000 AND POP_EST < 100000000"``
    bbox : tuple of (xmin, ymin, xmax, ymax), optional (default: None)
        If present, will be used to filter records whose geometry intersects this
        box.  This must be in the same CRS as the dataset.  If GEOS is present
        and used by GDAL, only geometries that intersect this bbox will be
        returned; if GEOS is not available or not used by GDAL, all geometries
        with bounding boxes that intersect this bbox will be returned.
        Cannot be combined with ``mask`` keyword.
    mask : Shapely geometry, optional (default: None)
        If present, will be used to filter records whose geometry intersects
        this geometry.  This must be in the same CRS as the dataset.  If GEOS is
        present and used by GDAL, only geometries that intersect this geometry
        will be returned; if GEOS is not available or not used by GDAL, all
        geometries with bounding boxes that intersect the bounding box of this
        geometry will be returned.  Requires Shapely >= 2.0.
        Cannot be combined with ``bbox`` keyword.
    fids : array-like, optional (default: None)
        Array of integer feature id (FID) values to select. Cannot be combined
        with other keywords to select a subset (``skip_features``,
        ``max_features``, ``where``, ``bbox``, or ``mask``). Note that the
        starting index is driver and file specific (e.g. typically 0 for
        Shapefile and 1 for GeoPackage, but can still depend on the specific
        file). The performance of reading a large number of features usings FIDs
        is also driver specific.
    sql : str, optional (default: None)
        The SQL statement to execute. Look at the sql_dialect parameter for more
        information on the syntax to use for the query. When combined with other
        keywords like ``columns``, ``skip_features``, ``max_features``,
        ``where``, ``bbox``, or ``mask``, those are applied after the SQL query.
        Be aware that this can have an impact on performance, (e.g. filtering
        with the ``bbox`` or ``mask`` keywords may not use spatial indexes).
        Cannot be combined with the ``layer`` or ``fids`` keywords.
    sql_dialect : str, optional (default: None)
        The SQL dialect the ``sql`` statement is written in. Possible values:

          - **None**: if the data source natively supports SQL, its specific SQL dialect
            will be used by default (eg. SQLite and Geopackage: `SQLITE`_, PostgreSQL).
            If the data source doesn't natively support SQL, the `OGRSQL`_ dialect is
            the default.
          - '`OGRSQL`_': can be used on any data source. Performance can suffer
            when used on data sources with native support for SQL.
          - '`SQLITE`_': can be used on any data source. All spatialite_
            functions can be used. Performance can suffer on data sources with
            native support for SQL, except for Geopackage and SQLite as this is
            their native SQL dialect.

    return_fids : bool, optional (default: False)
        If True, will return the FIDs of the feature that were read.
    datetime_as_string : bool, optional (default: False)
        If True, will return datetime dtypes as detected by GDAL as a string
        array (which can be used to extract time zone info), instead of
        a datetime64 array.
    datetime_layout : str, optional (default: None)
        If set to ``"utc_offset"``, DateTime fields are returned as two arrays:
        the field itself holds the exact instant in UTC (``datetime64[ms]``), and
        a companion ``<name>_offset`` field holds the original UTC offset as
        ``timedelta64[ms]``. A missing offset (``NaT``) means the file did not
        record a time zone, while zero means it recorded an explicit UTC offset.
        Cannot be combined with ``datetime_as_string``. The metadata is updated
        to describe both fields, and the offset fields have OGR type
        ``"OFTDateTimeOffset"``.

    **kwargs
        Additional driver-specific dataset open options passed to OGR.  Invalid
        options will trigger a warning.

    Returns
    -------
    (dict, fids, geometry, data fields)
        Returns a tuple of meta information about the data source in a dict,
        an ndarray of FIDs corresponding to the features that were read or None
        (if return_fids is False),
        an ndarray of geometry objects or None (if data source does not include
        geometry or read_geometry is False), a tuple of ndarrays for each field
        in the data layer.

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "dtypes": <ndarray of numpy dtypes corresponding to fields>,
            "ogr_types": <ndarray of OGR types corresponding to fields>,
            "ogr_subtypes": <ndarray of OGR subtypes corresponding to fields>,
            "encoding": "<encoding>",
            "geometry_type": "<geometry type>",
        }

    .. _OGRSQL:

        https://gdal.org/user/ogr_sql_dialect.html#ogr-sql-dialect

    .. _OGRSQL WHERE:

        https://gdal.org/user/ogr_sql_dialect.html#where

    .. _SQLITE:

        https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect

    .. _spatialite:

        https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html

    """
    dataset_kwargs = _preprocess_options_key_value(kwargs) if kwargs else {}
    _datetime.validate_layout(datetime_layout, datetime_as_string)

    meta, fids, geometry, field_data = ogr_read(
        get_vsi_path_or_buffer(path_or_buffer),
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        force_2d=force_2d,
        skip_features=skip_features,
        max_features=max_features or 0,
        where=where,
        bbox=bbox,
        mask=_mask_to_wkb(mask),
        fids=fids,
        sql=sql,
        sql_dialect=sql_dialect,
        return_fids=return_fids,
        dataset_kwargs=dataset_kwargs,
        datetime_as_string=datetime_as_string or _datetime.is_enabled(datetime_layout),
    )

    if _datetime.is_enabled(datetime_layout):
        meta, field_data = _add_utc_offset_columns(meta, field_data)

    return meta, fids, geometry, field_data


def read_arrow(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    force_2d=False,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    datetime_as_string=False,
    datetime_layout=None,
    **kwargs,
):
    """Read OGR data source into a pyarrow Table.

    See docstring of `read` for parameters.

    When ``datetime_layout="utc_offset"`` is used, DateTime columns are returned
    as a ``timestamp[ms, tz=UTC]`` column plus a ``duration[ms]``
    ``<name>_offset`` column (see `read`).

    Returns
    -------
    (dict, pyarrow.Table)

        Returns a tuple of meta information about the returned data in a dict,
        and a pyarrow Table with data.

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "dtypes": <ndarray of numpy dtypes corresponding to fields>,
            "ogr_types": <ndarray of OGR types corresponding to fields>,
            "ogr_subtypes": <ndarray of OGR subtypes corresponding to fields>,
            "encoding": "<encoding>",
            "geometry_type": "<geometry_type>",
            "geometry_name": "<name of geometry column in arrow table>",
            "fid_column": "<name of FID column in arrow table>"
        }

    """
    if not HAS_PYARROW:
        raise RuntimeError(
            "pyarrow required to read using 'read_arrow'. You can use 'open_arrow' "
            "to read data with an alternative Arrow implementation"
        )

    _datetime.validate_layout(datetime_layout, datetime_as_string)

    from pyarrow import Table

    gdal_version = get_gdal_version()

    if skip_features < 0:
        raise ValueError("'skip_features' must be >= 0")

    if max_features is not None and max_features < 0:
        raise ValueError("'max_features' must be >= 0")

    # remember the requested range before it is rearranged below
    requested_skip_features = skip_features
    requested_max_features = max_features

    # limit batch size to max_features if set
    if "batch_size" in kwargs:
        batch_size = kwargs.pop("batch_size")
    else:
        batch_size = 65_536

    if max_features is not None and max_features < batch_size:
        batch_size = max_features

    # handle skip_features internally within open_arrow if GDAL >= 3.8.0
    gdal_skip_features = 0
    if gdal_version >= (3, 8, 0):
        gdal_skip_features = skip_features
        skip_features = 0

    # The dual-column contract aligns the offsets read through the non-Arrow API
    # with the Arrow rows by FID, so the FID column is needed internally.
    dual_layout = _datetime.is_enabled(datetime_layout)
    internal_fids = return_fids or dual_layout
    user_wants_fids = return_fids

    with open_arrow(
        path_or_buffer,
        layer=layer,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        force_2d=force_2d,
        where=where,
        bbox=bbox,
        mask=mask,
        fids=fids,
        sql=sql,
        sql_dialect=sql_dialect,
        return_fids=internal_fids,
        skip_features=gdal_skip_features,
        batch_size=batch_size,
        use_pyarrow=True,
        datetime_as_string=datetime_as_string,
        **kwargs,
    ) as source:
        meta, reader = source

        if max_features is not None:
            batches = []
            count = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                    batches.append(batch)

                    count += len(batch)
                    if count >= (skip_features + max_features):
                        break

                except StopIteration:
                    break

            # use combine_chunks to release the original memory that included
            # too many features
            table = (
                Table.from_batches(batches, schema=reader.schema)
                .slice(skip_features, max_features)
                .combine_chunks()
            )

        elif skip_features > 0:
            table = reader.read_all().slice(skip_features).combine_chunks()

        else:
            table = reader.read_all()

    if dual_layout:
        meta, table = _apply_datetime_layout_to_arrow(
            path_or_buffer,
            meta,
            table,
            read_kwargs={
                "layer": layer,
                "encoding": encoding,
                "columns": columns,
                "skip_features": requested_skip_features,
                "max_features": requested_max_features,
                "where": where,
                "bbox": bbox,
                "mask": mask,
                "fids": fids,
                "sql": sql,
                "sql_dialect": sql_dialect,
                **kwargs,
            },
        )

        if not user_wants_fids:
            # the FID column was only read internally, so drop it again
            fid_column = meta.get("fid_column") or "OGC_FID"
            if fid_column in table.column_names:
                table = table.drop([fid_column])

    return meta, table


def open_arrow(
    path_or_buffer,
    /,
    layer=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    force_2d=False,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
    fids=None,
    sql=None,
    sql_dialect=None,
    return_fids=False,
    batch_size=65_536,
    use_pyarrow=False,
    datetime_as_string=False,
    datetime_layout=None,
    **kwargs,
):
    """Open OGR data source as a stream of Arrow record batches.

    See docstring of `read` for parameters.

    When ``datetime_layout="utc_offset"`` is used, the stream has each DateTime
    column replaced by a ``timestamp[ms, tz=UTC]`` column followed by a
    ``duration[ms]`` ``<name>_offset`` column. This requires ``pyarrow``, and each
    batch is converted independently, so an offset can never move to another row
    when the data is split into batches.
    The returned object is reading from a stream provided by OGR and must not be
    accessed after the OGR dataset has been closed, i.e. after the context manager has
    been closed.

    By default this functions returns a generic stream object implementing
    the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_stream__``
    method). This object can then be consumed by your Arrow implementation
    of choice that supports this protocol.
    Optionally, you can specify ``use_pyarrow=True`` to directly get the
    stream as a `pyarrow.RecordBatchReader`.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Other Parameters
    ----------------
    batch_size : int (default: 65_536)
        Maximum number of features to retrieve in a batch.
    use_pyarrow : bool (default: False)
        If True, return a pyarrow RecordBatchReader instead of a generic
        ArrowStream object. In the default case, this stream object needs
        to be passed to another library supporting the Arrow PyCapsule
        Protocol to consume the stream of data.
    datetime_as_string : bool, optional (default: False)
        If True, will return datetime dtypes as detected by GDAL as strings,
        as Arrow doesn't support e.g. mixed time zones.

    Examples
    --------
    >>> from pyogrio.raw import open_arrow
    >>> import pyarrow as pa
    >>> import shapely
    >>>
    >>> with open_arrow(path) as source:
    >>>     meta, stream = source
    >>>     # wrap the arrow stream object in a pyarrow RecordBatchReader
    >>>     reader = pa.RecordBatchReader.from_stream(stream)
    >>>     geom_col = meta["geometry_name"] or "wkb_geometry"
    >>>     for batch in reader:
    >>>         geometries = shapely.from_wkb(batch[geom_col])

    The returned `stream` object needs to be consumed by a library implementing
    the Arrow PyCapsule Protocol. In the above example, pyarrow is used through
    its RecordBatchReader. For this case, you can also specify ``use_pyarrow=True``
    to directly get this result as a short-cut:

    >>> with open_arrow(path, use_pyarrow=True) as source:
    >>>     meta, reader = source
    >>>     geom_col = meta["geometry_name"] or "wkb_geometry"
    >>>     for batch in reader:
    >>>         geometries = shapely.from_wkb(batch[geom_col])

    Returns
    -------
    (dict, pyarrow.RecordBatchReader or ArrowStream)

        Returns a tuple of meta information about the data source in a dict,
        and a data stream object (a generic ArrowStream object, or a pyarrow
        RecordBatchReader if `use_pyarrow` is set to True).

        Meta is: {
            "crs": "<crs>",
            "fields": <ndarray of field names>,
            "dtypes": <ndarray of numpy dtypes corresponding to fields>,
            "ogr_types": <ndarray of OGR types corresponding to fields>,
            "ogr_subtypes": <ndarray of OGR subtypes corresponding to fields>,
            "encoding": "<encoding>",
            "geometry_type": "<geometry_type>",
            "geometry_name": "<name of geometry column in arrow table>",
            "fid_column": "<name of FID column in arrow table>"
        }

    """
    dataset_kwargs = _preprocess_options_key_value(kwargs) if kwargs else {}
    _datetime.validate_layout(datetime_layout, datetime_as_string)

    if not _datetime.is_enabled(datetime_layout):
        return ogr_open_arrow(
            get_vsi_path_or_buffer(path_or_buffer),
            layer=layer,
            encoding=encoding,
            columns=columns,
            read_geometry=read_geometry,
            force_2d=force_2d,
            skip_features=skip_features,
            max_features=max_features or 0,
            where=where,
            bbox=bbox,
            mask=_mask_to_wkb(mask),
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
            dataset_kwargs=dataset_kwargs,
            batch_size=batch_size,
            use_pyarrow=use_pyarrow,
            datetime_as_string=datetime_as_string,
        )

    # The dual-column contract needs the ISO8601 representation of each
    # date-time, which the Arrow stream only provides from GDAL 3.11 on (and
    # then offsets are attached per row while each batch is converted on its
    # own, so they cannot shift).
    if get_gdal_version() < (3, 11, 0):
        raise NotImplementedError(
            "'datetime_layout=\"utc_offset\"' through open_arrow requires GDAL "
            ">= 3.11; use read_arrow() or the non-Arrow API instead"
        )

    return _DatetimeLayoutContext(
        ogr_open_arrow(
            get_vsi_path_or_buffer(path_or_buffer),
            layer=layer,
            encoding=encoding,
            columns=columns,
            read_geometry=read_geometry,
            force_2d=force_2d,
            skip_features=skip_features,
            max_features=max_features or 0,
            where=where,
            bbox=bbox,
            mask=_mask_to_wkb(mask),
            fids=fids,
            sql=sql,
            sql_dialect=sql_dialect,
            return_fids=return_fids,
            dataset_kwargs=dataset_kwargs,
            batch_size=batch_size,
            use_pyarrow=use_pyarrow,
            datetime_as_string=True,
        ),
        use_pyarrow=use_pyarrow,
    )


class _DatetimeLayoutContext:
    """Context manager returned by ``open_arrow(datetime_layout="utc_offset")``."""

    def __init__(self, source_context, use_pyarrow):
        self._source_context = source_context
        self._use_pyarrow = use_pyarrow
        self._source = None

    def __enter__(self):
        self._source = self._source_context.__enter__()
        meta, reader = self._source
        new_meta = _new_meta_for_datetime_layout(meta)
        converted = _DatetimeLayoutReader(reader, meta)
        if self._use_pyarrow:
            import pyarrow as pa

            converted = pa.RecordBatchReader.from_stream(converted.__arrow_c_stream__())
        return new_meta, converted

    def __exit__(self, *exc_info):
        return self._source_context.__exit__(*exc_info)


class _DatetimeLayoutReader:
    """Arrow stream that converts each batch individually and then yields it.

    Converting batch by batch keeps every offset on its own row no matter how the
    data is split.
    """

    def __init__(self, reader, meta):
        self._reader = reader
        self._meta = meta
        self._datetime_names = _datetime_field_names(meta)

    def __arrow_c_stream__(self, requested_schema=None):
        table = self._build_table()
        return table.__arrow_c_stream__(requested_schema)

    def _build_table(self):
        import pyarrow as pa

        batches = (self._convert(batch) for batch in self._reader)
        return pa.Table.from_batches(batches)

    def _convert(self, batch):
        import pyarrow as pa

        replacements = {}
        for name in self._datetime_names:
            values = batch.column(name).to_pylist()
            utc, offsets = _datetime.parse_datetime_array(values, name)
            replacements[name] = [
                (name, pa.array(utc).cast(pa.timestamp("ms", tz="UTC"))),
                (_datetime.offset_column_name(name), pa.array(offsets)),
            ]

        output = []
        for name in batch.schema.names:
            entries = replacements.get(name)
            if entries is not None:
                output.extend(entries)
            else:
                output.append((name, batch.column(name)))
        return pa.RecordBatch.from_arrays(
            [array for _, array in output], names=[name for name, _ in output]
        )


def _parse_options_names(xml):
    """Convert metadata xml to list of names."""  # Based on Fiona's meta.py
    # (https://github.com/Toblerity/Fiona/blob/91c13ad8424641557a4e5f038f255f9b657b1bc5/fiona/meta.py)
    import xml.etree.ElementTree as ET

    options = []
    if xml:
        root = ET.fromstring(xml)
        for option in root.iter("Option"):
            # some options explicitly have scope='raster'
            if option.attrib.get("scope", "vector") != "raster":
                options.append(option.attrib["name"])

    return options


def _validate_metadata(dataset_metadata, layer_metadata, metadata):
    """Validate the metadata."""
    if metadata is not None:
        if layer_metadata is not None:
            raise ValueError("Cannot pass both metadata and layer_metadata")
        layer_metadata = metadata

    # validate metadata types
    for meta in [dataset_metadata, layer_metadata]:
        if meta is not None:
            for k, v in meta.items():
                if not isinstance(k, str):
                    raise ValueError(f"metadata key {k} must be a string")

                if not isinstance(v, str):
                    raise ValueError(f"metadata value {v} must be a string")

    return dataset_metadata, layer_metadata


def _preprocess_options_kwargs(driver, dataset_options, layer_options, kwargs):
    """Preprocess kwargs and split in dataset and layer creation options."""
    dataset_kwargs = _preprocess_options_key_value(dataset_options or {})
    layer_kwargs = _preprocess_options_key_value(layer_options or {})
    if kwargs:
        kwargs = _preprocess_options_key_value(kwargs)
        dataset_option_names = _parse_options_names(
            _get_driver_metadata_item(driver, "DMD_CREATIONOPTIONLIST")
        )
        layer_option_names = _parse_options_names(
            _get_driver_metadata_item(driver, "DS_LAYER_CREATIONOPTIONLIST")
        )
        for k, v in kwargs.items():
            if k in dataset_option_names:
                dataset_kwargs[k] = v
            elif k in layer_option_names:
                layer_kwargs[k] = v
            else:
                raise ValueError(f"unrecognized option '{k}' for driver '{driver}'")

    return dataset_kwargs, layer_kwargs


def _get_write_path_driver(path, driver, append=False):
    """Validate and return path and driver.

    Parameters
    ----------
    path : str or io.BytesIO
        path to output file on writeable file system or an io.BytesIO object to
        allow writing to memory.  Will raise NotImplementedError if an open file
        handle is passed.
    driver : str, optional (default: None)
        The OGR format driver used to write the vector file. By default attempts
        to infer driver from path.  Must be provided to write to a file-like
        object.
    append : bool, optional (default: False)
        True if path and driver is being tested for append support

    Returns
    -------
    (path, driver)

    """
    if isinstance(path, BytesIO):
        if driver is None:
            raise ValueError("driver must be provided to write to in-memory file")

        # blacklist certain drivers known not to work in current memory implementation
        # because they create multiple files
        if driver in {"ESRI Shapefile", "OpenFileGDB"}:
            raise ValueError(f"writing to in-memory file is not supported for {driver}")

        # verify that driver supports VSI methods
        if not ogr_driver_supports_vsi(driver):
            raise DataSourceError(
                f"{driver} does not support ability to write in-memory in GDAL "
                f"{get_gdal_version_string()}"
            )

        if append:
            raise NotImplementedError("append is not supported for in-memory files")

    elif hasattr(path, "write") and not isinstance(path, Path):
        raise NotImplementedError(
            "writing to an open file handle is not yet supported; instead, write to a "
            "BytesIO instance and then read bytes from that to write to the file handle"
        )

    else:
        path = vsi_path(path)

        if driver is None:
            driver = detect_write_driver(path)

    # verify that driver supports writing
    if not ogr_driver_supports_write(driver):
        raise DataSourceError(
            f"{driver} does not support write functionality in GDAL "
            f"{get_gdal_version_string()}"
        )

    return path, driver


def _prepare_datetime_layout_write(
    driver, fields, field_data, field_mask, gdal_tz_offsets, datetime_layout
):
    """Turn UTC + offset column pairs back into GDAL date-time fields.

    Returns the remaining fields, field data and field masks, together with the
    computed ``gdal_tz_offsets``.  The ``<name>_offset`` columns are removed.
    """
    if not _datetime.is_enabled(datetime_layout):
        return field_data, fields, field_mask, gdal_tz_offsets

    offset_columns = _datetime.find_offset_columns(
        fields,
        dtypes_by_name={
            name: getattr(values, "dtype", None)
            for name, values in zip(fields, field_data, strict=True)
        },
    )
    if not offset_columns:
        return field_data, fields, field_mask, gdal_tz_offsets

    _datetime.validate_driver_supports_datetime_offset(driver)

    positions = {name: index for index, name in enumerate(fields)}
    drop = {positions[offset] for offset in offset_columns.values()}
    combined = dict(gdal_tz_offsets or {})

    for name, offset_name in offset_columns.items():
        utc_values = field_data[positions[name]]
        offset_values = field_data[positions[offset_name]]
        wall, flags = _datetime.prepare_write(utc_values, offset_values, name)
        field_data[positions[name]] = wall
        combined[name] = flags

    def keep(values):
        if values is None:
            return None
        return [value for index, value in enumerate(values) if index not in drop]

    return keep(field_data), keep(fields), keep(field_mask), combined


def write(
    path,
    geometry,
    field_data,
    fields,
    field_mask=None,
    layer=None,
    driver=None,
    # derived from meta if roundtrip
    geometry_type=None,
    crs=None,
    encoding=None,
    promote_to_multi=None,
    nan_as_null=True,
    append=False,
    dataset_metadata=None,
    layer_metadata=None,
    metadata=None,
    dataset_options=None,
    layer_options=None,
    gdal_tz_offsets=None,
    datetime_layout=None,
    **kwargs,
):
    """Write geometry and field data to an OGR file format.

    Parameters
    ----------
    path : str or io.BytesIO
        path to output file on writeable file system or an io.BytesIO object to
        allow writing to memory.  Will raise NotImplementedError if an open file
        handle is passed; use BytesIO instead.
        NOTE: support for writing to memory is limited to specific drivers.
    geometry : ndarray of WKB encoded geometries or None
        If None, geometries will not be written to output file
    field_data : list-like of shape (num_fields, num_records)
        contains one record per field to be written in same order as fields
    fields : list-like
        contains field names
    field_mask : list-like of ndarrays or None, optional (default: None)
        contains mask arrays indicating null values of the field at the same
        position in the outer list, or None to indicate field does not have
        a mask array
    layer : str, optional (default: None)
        layer name to create.  If writing to memory and layer name is not
        provided, it layer name will be set to a UUID4 value.
    driver : string, optional (default: None)
        The OGR format driver used to write the vector file. By default attempts
        to infer driver from path.  Must be provided to write to memory.
    geometry_type : str, optional (default: None)
        Possible values are: "Unknown", "Point", "LineString", "Polygon",
        "MultiPoint", "MultiLineString", "MultiPolygon" or "GeometryCollection".

        This parameter does not modify the geometry, but it will try to force
        the layer type of the output file to this value. Use this parameter with
        caution because using a wrong layer geometry type may result in errors
        when writing the file, may be ignored by the driver, or may result in
        invalid files.
    crs : str, optional (default: None)
        WKT-encoded CRS of the geometries to be written.
    encoding : str, optional (default: None)
        If present, will be used as the encoding for writing string values to
        the file.  Use with caution, only certain drivers support encodings
        other than UTF-8.
    promote_to_multi : bool, optional (default: None)
        If True, will convert singular geometry types in the data to their
        corresponding multi geometry type for writing. By default, will convert
        mixed singular and multi geometry types to multi geometry types for
        drivers that do not support mixed singular and multi geometry types. If
        False, geometry types will not be promoted, which may result in errors
        or invalid files when attempting to write mixed singular and multi
        geometry types to drivers that do not support such combinations.
    nan_as_null : bool, default True
        For floating point columns (float32 / float64), whether NaN values are
        written as "null" (missing value). Defaults to True because in pandas
        NaNs are typically used as missing value. Note that when set to False,
        behaviour is format specific: some formats don't support NaNs by
        default (e.g. GeoJSON will skip this property) or might treat them as
        null anyway (e.g. GeoPackage).
    append : bool, optional (default: False)
        If True, the data source specified by path already exists, and the
        driver supports appending to an existing data source, will cause the
        data to be appended to the existing records in the data source.  Not
        supported for writing to in-memory files.
        NOTE: append support is limited to specific drivers and GDAL versions.
    dataset_metadata : dict, optional (default: None)
        Metadata to be stored at the dataset level in the output file; limited
        to drivers that support writing metadata, such as GPKG, and silently
        ignored otherwise. Keys and values must be strings.
    layer_metadata : dict, optional (default: None)
        Metadata to be stored at the layer level in the output file; limited to
        drivers that support writing metadata, such as GPKG, and silently
        ignored otherwise. Keys and values must be strings.
    metadata : dict, optional (default: None)
        alias of layer_metadata
    dataset_options : dict, optional
        Dataset creation options (format specific) passed to OGR. Specify as
        a key-value dictionary.
    layer_options : dict, optional
        Layer creation options (format specific) passed to OGR. Specify as
        a key-value dictionary.
    gdal_tz_offsets : dict, optional (default: None)
        Used to handle GDAL time zone offsets for each field contained in dict.
    datetime_layout : str, optional (default: None)
        If set to ``"utc_offset"``, each datetime64 field that has a
        ``<name>_offset`` timedelta64 field next to it is written back as one
        date-time field: the UTC value and its offset are combined into the local
        wall clock value that GDAL stores, with the offset written alongside it.
        The offset field itself is not written, and its name is removed from
        ``fields``. ``NaT`` offsets are written as an unknown time zone. Values
        that GDAL cannot store exactly (sub-millisecond date-times or offsets that
        are not a whole multiple of 15 minutes, larger than +14:00 or smaller
        than -12:00) raise ``ValueError`` instead of being rounded.
    **kwargs
        Additional driver-specific dataset creation options passed to OGR. Invalid
        options will trigger a warning.

    """
    # remove some unneeded kwargs (e.g. dtypes is included in meta returned by
    # read, and it is convenient to pass meta directly into write for round trip tests)
    kwargs.pop("dtypes", None)
    kwargs.pop("ogr_types", None)
    kwargs.pop("ogr_subtypes", None)

    path, driver = _get_write_path_driver(path, driver, append=append)

    dataset_metadata, layer_metadata = _validate_metadata(
        dataset_metadata, layer_metadata, metadata
    )

    if geometry is not None and promote_to_multi is None:
        promote_to_multi = (
            geometry_type.startswith("Multi")
            and driver in DRIVERS_NO_MIXED_SINGLE_MULTI
        )

    if geometry is not None and crs is None:
        warnings.warn(
            "'crs' was not provided.  The output dataset will not have "
            "projection information defined and may not be usable in other "
            "systems.",
            stacklevel=2,
        )

    # preprocess kwargs and split in dataset and layer creation options
    dataset_kwargs, layer_kwargs = _preprocess_options_kwargs(
        driver, dataset_options, layer_options, kwargs
    )

    if field_data is not None and fields is not None:
        field_data, fields, field_mask, gdal_tz_offsets = (
            _prepare_datetime_layout_write(
                driver,
                list(fields),
                list(field_data),
                None if field_mask is None else list(field_mask),
                gdal_tz_offsets,
                datetime_layout,
            )
        )

    ogr_write(
        path,
        layer=layer,
        driver=driver,
        geometry=geometry,
        geometry_type=geometry_type,
        field_data=list(field_data) if field_data is not None else None,
        field_mask=list(field_mask) if field_mask is not None else None,
        fields=list(fields) if fields is not None else None,
        crs=crs,
        encoding=encoding,
        promote_to_multi=promote_to_multi,
        nan_as_null=nan_as_null,
        append=append,
        dataset_metadata=dataset_metadata,
        layer_metadata=layer_metadata,
        dataset_kwargs=dataset_kwargs,
        layer_kwargs=layer_kwargs,
        gdal_tz_offsets=gdal_tz_offsets,
    )


def _arrow_combine_datetime_layout(arrow_obj, driver):
    """Combine UTC + offset column pairs of an Arrow table for writing.

    A ``timestamp`` + ``duration`` pair cannot be written as one date-time field
    through GDAL's Arrow writer, so the pair is turned into an ISO8601 string
    column (with a per-row offset) and marked as a DateTime field via column
    metadata, which GDAL understands.
    """
    if not HAS_PYARROW:
        raise RuntimeError("pyarrow is required to write with 'datetime_layout'")

    import pyarrow as pa

    capsule = arrow_obj.__arrow_c_stream__()
    table = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
    names = table.column_names
    # only treat ``<name>_offset`` as a companion when it holds a duration and
    # ``<name>`` holds a timestamp, so an unrelated column of the same name is
    # left alone
    offset_names = {
        name
        for name in names
        if _datetime.is_offset_column_name(name)
        and pa.types.is_duration(table.schema.field(name).type)
        and _datetime.base_column_name(name) in names
        and pa.types.is_timestamp(
            table.schema.field(_datetime.base_column_name(name)).type
        )
    }
    if not offset_names:
        return table

    _datetime.validate_driver_supports_datetime_offset(driver)

    datetime_columns = []
    replacements = {}
    for offset_name in offset_names:
        name = _datetime.base_column_name(offset_name)
        if name not in names:
            continue
        timestamp_field = table.schema.field(name)
        column = table.column(name)
        if timestamp_field.type.tz not in (None, "UTC"):
            # bring any other time zone to UTC first; the exact instant is
            # unchanged and the original offset is in the companion column
            column = column.cast(pa.timestamp("ms", tz="UTC"))
        utc_values = column.cast(pa.timestamp("ms")).to_numpy(zero_copy_only=False)
        offsets = table.column(offset_name).to_numpy(zero_copy_only=False)
        _datetime.utc_to_wall_and_flags(utc_values, offsets, name)
        strings = _datetime.format_utc_offset_strings(utc_values, offsets)
        replacements[name] = (name, pa.array(strings, type=pa.string()))
        datetime_columns.append(name)

    if not replacements:
        return table

    output = []
    for name in names:
        if name in offset_names:
            continue
        replacement = replacements.get(name)
        output.append(
            replacement if replacement is not None else (name, table.column(name))
        )

    result = pa.table(dict(output))
    fields = []
    for field in result.schema:
        if field.name in datetime_columns:
            metadata = dict(field.metadata or {})
            metadata[b"GDAL:OGR:type"] = b"DateTime"
            fields.append(field.with_metadata(metadata))
        else:
            fields.append(field)
    return result.cast(pa.schema(fields, metadata=result.schema.metadata))


def write_arrow(
    arrow_obj,
    path,
    layer=None,
    driver=None,
    geometry_name=None,
    geometry_type=None,
    crs=None,
    encoding=None,
    append=False,
    dataset_metadata=None,
    layer_metadata=None,
    metadata=None,
    dataset_options=None,
    layer_options=None,
    datetime_layout=None,
    **kwargs,
):
    """Write an Arrow-compatible data source to an OGR file format.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Parameters
    ----------
    arrow_obj
        The Arrow data to write. This can be any Arrow-compatible tabular data
        object that implements the `Arrow PyCapsule Protocol`_ (i.e. has an
        ``__arrow_c_stream__`` method), for example a pyarrow Table or
        RecordBatchReader.
    path : str or io.BytesIO
        path to output file on writeable file system or an io.BytesIO object to
        allow writing to memory
        NOTE: support for writing to memory is limited to specific drivers.
    layer : str, optional (default: None)
        layer name to create.  If writing to memory and layer name is not
        provided, it layer name will be set to a UUID4 value.
    driver : string, optional (default: None)
        The OGR format driver used to write the vector file. By default attempts
        to infer driver from path.  Must be provided to write to memory.
    geometry_name : str, optional (default: None)
        The name of the column in the input data that will be written as the
        geometry field. Will be inferred from the input data if the geometry
        column is annotated as an "geoarrow.wkb" or "ogc.wkb" extension type.
        Otherwise needs to be specified explicitly.
    geometry_type : str
        The geometry type of the written layer. Currently, this needs to be
        specified explicitly when creating a new layer with geometries.
        Possible values are: "Unknown", "Point", "LineString", "Polygon",
        "MultiPoint", "MultiLineString", "MultiPolygon" or "GeometryCollection".

        This parameter does not modify the geometry, but it will try to force the layer
        type of the output file to this value. Use this parameter with caution because
        using a wrong layer geometry type may result in errors when writing the
        file, may be ignored by the driver, or may result in invalid files.
    crs : str, optional (default: None)
        WKT-encoded CRS of the geometries to be written.
    encoding : str, optional (default: None)
        Only used for the .dbf file of ESRI Shapefiles. If not specified,
        uses the default locale.
    append : bool, optional (default: False)
        If True, the data source specified by path already exists, and the
        driver supports appending to an existing data source, will cause the
        data to be appended to the existing records in the data source.  Not
        supported for writing to in-memory files.
        NOTE: append support is limited to specific drivers and GDAL versions.
    dataset_metadata : dict, optional (default: None)
        Metadata to be stored at the dataset level in the output file; limited
        to drivers that support writing metadata, such as GPKG, and silently
        ignored otherwise. Keys and values must be strings.
    layer_metadata : dict, optional (default: None)
        Metadata to be stored at the layer level in the output file; limited to
        drivers that support writing metadata, such as GPKG, and silently
        ignored otherwise. Keys and values must be strings.
    metadata : dict, optional (default: None)
        alias of layer_metadata
    dataset_options : dict, optional
        Dataset creation options (format specific) passed to OGR. Specify as
        a key-value dictionary.
    layer_options : dict, optional
        Layer creation options (format specific) passed to OGR. Specify as
        a key-value dictionary.
    datetime_layout : str, optional (default: None)
        If set to ``"utc_offset"``, each ``timestamp`` column ``<name>`` that has
        a ``duration`` column ``<name>_offset`` next to it is written back as one
        date-time field: the local wall clock time is reconstructed from the UTC
        value and the offset, and the offset column is dropped. ``NaT`` offsets
        are written as an unknown time zone. Drivers that cannot store date-time
        fields are rejected, as are values that GDAL cannot store exactly.
    **kwargs
        Additional driver-specific dataset or layer creation options passed
        to OGR. pyogrio will attempt to automatically pass those keywords
        either as dataset or as layer creation option based on the known
        options for the specific driver. Alternatively, you can use the
        explicit `dataset_options` or `layer_options` keywords to manually
        do this (for example if an option exists as both dataset and layer
        option).

    """
    if not HAS_ARROW_WRITE_API:
        raise RuntimeError("GDAL>=3.8 required to write using arrow")

    if not hasattr(arrow_obj, "__arrow_c_stream__"):
        raise ValueError(
            "The provided data is not recognized as Arrow data. The object "
            "should implement the Arrow PyCapsule Protocol (i.e. have a "
            "'__arrow_c_stream__' method)."
        )

    _datetime.validate_layout(datetime_layout)

    path, driver = _get_write_path_driver(path, driver, append=append)

    if _datetime.is_enabled(datetime_layout):
        arrow_obj = _arrow_combine_datetime_layout(arrow_obj, driver)

    if "promote_to_multi" in kwargs:
        raise ValueError(
            "The 'promote_to_multi' option is not supported when writing using Arrow"
        )

    if geometry_name is not None:
        if geometry_type is None:
            raise ValueError("'geometry_type' keyword is required")
        if crs is None:
            # TODO: does GDAL infer CRS automatically from geometry metadata?
            warnings.warn(
                "'crs' was not provided.  The output dataset will not have "
                "projection information defined and may not be usable in other "
                "systems.",
                stacklevel=2,
            )

    dataset_metadata, layer_metadata = _validate_metadata(
        dataset_metadata, layer_metadata, metadata
    )

    # preprocess kwargs and split in dataset and layer creation options
    dataset_kwargs, layer_kwargs = _preprocess_options_kwargs(
        driver, dataset_options, layer_options, kwargs
    )

    ogr_write_arrow(
        path,
        layer=layer,
        driver=driver,
        arrow_obj=arrow_obj,
        geometry_type=geometry_type,
        geometry_name=geometry_name,
        crs=crs,
        encoding=encoding,
        append=append,
        dataset_metadata=dataset_metadata,
        layer_metadata=layer_metadata,
        dataset_kwargs=dataset_kwargs,
        layer_kwargs=layer_kwargs,
    )
