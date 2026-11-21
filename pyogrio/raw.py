"""Low level functions to read and write OGR data sources."""

import contextlib
import warnings
from io import BytesIO
from pathlib import Path

from pyogrio._compat import HAS_ARROW_WRITE_API, HAS_PYARROW
from pyogrio._env import GDALEnv
from pyogrio.core import detect_write_driver, list_layers
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

    return ogr_read(
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
        datetime_as_string=datetime_as_string,
    )


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
    **kwargs,
):
    """Read OGR data source into a pyarrow Table.

    See docstring of `read` for parameters.

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

    from pyarrow import Table

    gdal_version = get_gdal_version()

    if skip_features < 0:
        raise ValueError("'skip_features' must be >= 0")

    if max_features is not None and max_features < 0:
        raise ValueError("'max_features' must be >= 0")

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
        return_fids=return_fids,
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
    **kwargs,
):
    """Open OGR data source as a stream of Arrow record batches.

    See docstring of `read` for parameters.

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


#: Name of the column carrying the originating layer of each feature.
SOURCE_LAYER_COLUMN = "source_layer"
#: Name of the column carrying the original FID (per layer) of each feature.
SOURCE_FID_COLUMN = "source_fid"
#: Name of the geometry column in the combined output.
GEOMETRY_COLUMN = "geometry"

_RESERVED_COLUMNS = {SOURCE_LAYER_COLUMN, SOURCE_FID_COLUMN}


def read_arrow_layers(
    path_or_buffer,
    /,
    layers=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
    fids=None,
    datetime_as_string=False,
    **kwargs,
):
    """Read multiple layers of an OGR data source into one pyarrow Table.

    The features of the requested layers are combined into a single table, in
    the declared order of ``layers``, so that several source layers become one
    vector stream. Every row records the layer it came from in a
    ``source_layer`` column and its original feature id (FID) in a
    ``source_fid`` column. Because the source layer is part of every row,
    features that share an FID in different layers remain distinct records.

    Attribute fields with the same name across layers are aligned to a common
    Arrow schema. Where layers disagree on the Arrow type of a shared field, the
    types are unified using the "permissive" promotion rules of
    :func:`pyarrow.unify_schemas` (for example ``int32`` and ``int64`` become
    ``int64``). A field that is absent from a given layer is filled with nulls
    for that layer's rows.

    Layers without a geometry column are supported: when at least one selected
    layer has geometry, the combined table has a single ``geometry`` column
    whose values are null for the rows of geometry-less layers. When none of the
    selected layers has geometry, the combined table has no geometry column.

    Features are read and emitted one batch at a time; layers are never first
    combined per layer into a single large table.

    See the docstring of `read` for the shared parameters, and
    :func:`open_arrow_layers` for the streaming equivalent.

    Parameters
    ----------
    path_or_buffer : pathlib.Path or str, or bytes buffer
        A dataset path or URI, raw buffer, or file-like object with a read method.
    layers : sequence of str or int, optional (default: all layers)
        Names or indices, in the order in which they should be read, of the
        layers to combine. When omitted, every layer of the data source is read
        in its declared order. Indices refer to the position of the layer in the
        data source.
    encoding, columns, read_geometry, skip_features, max_features, where, bbox, \
mask, fids, datetime_as_string, **kwargs
        See the docstring of `read` for these parameters. ``columns`` and
        ``fids`` apply to every selected layer (FIDs are filtered per layer,
        within each layer's own FID space). ``skip_features`` and
        ``max_features`` apply per layer.

    Returns
    -------
    (dict, pyarrow.Table)
        Returns a tuple of meta information and a pyarrow Table with the
        combined data. Meta is::

            {
                "layers": <list of layer names in read order>,
                "geometry_name": "<geometry column name or None>",
                "geometry_type": "<common geometry type or 'Unknown'>",
                "crs": "<CRS shared by all spatial layers, or None>",
                "source_layer_column": "source_layer",
                "source_fid_column": "source_fid",
                "fields": <list of combined attribute field names>,
                "layer_meta": <list of per-layer meta dicts as returned by
                               open_arrow>,
            }

    Raises
    ------
    ValueError
        If the selected layers have incompatible CRS, if an attribute field
        cannot be aligned to a common Arrow type, or if a source attribute uses
        one of the reserved output column names.

    """
    if not HAS_PYARROW:
        raise RuntimeError(
            "pyarrow is required to read with 'read_arrow_layers'; use "
            "'open_arrow_layers' to obtain a generic Arrow stream"
        )

    with open_arrow_layers(
        path_or_buffer,
        layers=layers,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        skip_features=skip_features,
        max_features=max_features,
        where=where,
        bbox=bbox,
        mask=mask,
        fids=fids,
        datetime_as_string=datetime_as_string,
        use_pyarrow=True,
        **kwargs,
    ) as (meta, reader):
        return meta, reader.read_all()


def open_arrow_layers(
    path_or_buffer,
    /,
    layers=None,
    encoding=None,
    columns=None,
    read_geometry=True,
    skip_features=0,
    max_features=None,
    where=None,
    bbox=None,
    mask=None,
    fids=None,
    datetime_as_string=False,
    batch_size=65_536,
    use_pyarrow=False,
    **kwargs,
):
    """Open multiple layers of an OGR data source as one Arrow record batch stream.

    The stream concatenates the features of the requested layers, in the
    declared order of ``layers``, so that several source layers become a single
    vector stream. Every row records its originating layer in a ``source_layer``
    column and its original feature id in a ``source_fid`` column, so identical
    FIDs from different layers stay distinct records.

    Attribute fields that share a name across layers are aligned to a common
    Arrow schema (see :func:`read_arrow_layers`). Geometry-less layers and
    layers with missing fields are supported by filling the corresponding
    values with nulls.

    Layers are opened and consumed one batch at a time. Features of a layer are
    streamed before the next layer is opened, and layers are never combined into
    a single in-memory table.

    See the docstring of :func:`read_arrow_layers` and `read` for parameters.

    The returned stream reads from OGR and must not be accessed after the
    context manager has been closed.

    By default this function returns a generic stream object implementing the
    `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_stream__`` method)
    that can be consumed by any Arrow implementation supporting that protocol.
    Specify ``use_pyarrow=True`` to directly get a `pyarrow.RecordBatchReader`.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Other Parameters
    ----------------
    batch_size : int (default: 65_536)
        Maximum number of features to retrieve per batch.
    use_pyarrow : bool (default: False)
        If True, return a pyarrow RecordBatchReader instead of a generic
        ArrowStream object.

    Returns
    -------
    (dict, pyarrow.RecordBatchReader or ArrowStream)
        The same metadata as returned by `read_arrow_layers`, and a stream of
        record batches.

    Examples
    --------
    >>> from pyogrio.raw import open_arrow_layers
    >>> with open_arrow_layers(path, layers=["roads", "rivers"]) as (meta, stream):
    >>>     for batch in stream:
    >>>         ...  # batch has "source_layer", "source_fid" and aligned fields

    """
    if not HAS_ARROW_WRITE_API:
        raise RuntimeError("GDAL>=3.8 is required to read Arrow streams per layer")

    if not HAS_PYARROW:
        raise RuntimeError(
            "pyarrow is required to align multiple layers to a common Arrow schema"
        )

    if skip_features < 0:
        raise ValueError("'skip_features' must be >= 0")

    if max_features is not None and max_features < 0:
        raise ValueError("'max_features' must be >= 0")

    # Normalize the input once so that file-like buffers are read only once and
    # the same (VSI) path is reused for every layer that is opened below.
    path_or_buffer = get_vsi_path_or_buffer(path_or_buffer)

    declared = _list_layer_names(path_or_buffer)

    if layers is None:
        layer_names = declared
    else:
        if isinstance(layers, (str, int)) and not isinstance(layers, bool):
            raise TypeError(
                "'layers' must be a sequence of layer names or indices, not a "
                f"single {type(layers).__name__}"
            )
        layer_names = [_resolve_layer_name(item, declared) for item in layers]

    if len(layer_names) == 0:
        raise ValueError("'layers' must contain at least one layer")

    if len(set(layer_names)) != len(layer_names):
        raise ValueError(f"Selected layers must be unique: {layer_names}")

    return _open_arrow_layers(
        path_or_buffer,
        layer_names,
        encoding=encoding,
        columns=columns,
        read_geometry=read_geometry,
        skip_features=skip_features,
        max_features=max_features,
        where=where,
        bbox=bbox,
        mask=mask,
        fids=fids,
        datetime_as_string=datetime_as_string,
        batch_size=batch_size,
        use_pyarrow=use_pyarrow,
        kwargs=kwargs,
    )


def _resolve_layer_name(layer, declared):
    """Resolve a layer name or index to its declared layer name."""
    if isinstance(layer, str):
        if layer not in declared:
            raise ValueError(
                f"Layer {layer!r} not found in data source; available layers: "
                f"{declared}"
            )
        return layer

    if isinstance(layer, int) and not isinstance(layer, bool):
        try:
            return declared[layer]
        except IndexError:
            raise ValueError(
                f"Layer index {layer} is out of range for data source with "
                f"{len(declared)} layer(s)"
            ) from None

    raise TypeError(
        "Each entry in 'layers' must be a layer name (str) or index (int), "
        f"got {type(layer).__name__}"
    )


def _list_layer_names(path_or_buffer):
    """Return the declared layer names of a data source, in source order."""
    return [name for name, _geometry_type in list_layers(path_or_buffer)]


@contextlib.contextmanager
def _open_arrow_layers(
    path_or_buffer,
    layer_names,
    *,
    encoding,
    columns,
    read_geometry,
    skip_features,
    max_features,
    where,
    bbox,
    mask,
    fids,
    datetime_as_string,
    batch_size,
    use_pyarrow,
    kwargs,
):
    """Implementation of :func:`open_arrow_layers`."""
    import pyarrow as pa

    open_kwargs = {
        "encoding": encoding,
        "columns": columns,
        "read_geometry": read_geometry,
        "skip_features": skip_features,
        "where": where,
        "bbox": bbox,
        "mask": mask,
        "fids": fids,
        "datetime_as_string": datetime_as_string,
        "batch_size": batch_size,
        "use_pyarrow": True,
    }

    # Inspect every layer to construct the common output schema. Only the
    # schema is retained; the per-layer streams are closed again immediately.
    layer_info = []
    for layer in layer_names:
        with open_arrow(
            path_or_buffer, layer=layer, return_fids=True, **open_kwargs, **kwargs
        ) as (meta, reader):
            layer_info.append(_LayerArrowInfo(layer, meta, reader.schema))

    geometry_field, geometry_type, crs = _validate_layer_geometry(layer_info)
    attribute_fields = _unify_attribute_fields(layer_info)
    _check_reserved_columns(attribute_fields, geometry_field)

    fields = [
        pa.field(SOURCE_LAYER_COLUMN, pa.string(), nullable=False),
        pa.field(SOURCE_FID_COLUMN, pa.int64(), nullable=False),
        *attribute_fields,
    ]
    if geometry_field is not None:
        fields.append(geometry_field)

    output_schema = pa.schema(fields)

    meta = {
        "layers": [info.name for info in layer_info],
        "geometry_name": GEOMETRY_COLUMN if geometry_field is not None else None,
        "geometry_type": geometry_type,
        "crs": crs,
        "source_layer_column": SOURCE_LAYER_COLUMN,
        "source_fid_column": SOURCE_FID_COLUMN,
        "fields": [field.name for field in attribute_fields],
        "layer_meta": [info.meta for info in layer_info],
    }

    reader = pa.RecordBatchReader.from_batches(
        output_schema,
        _iter_aligned_batches(
            path_or_buffer,
            layer_info,
            attribute_fields,
            geometry_field,
            output_schema,
            open_kwargs,
            kwargs,
            max_features,
        ),
    )

    try:
        if use_pyarrow:
            yield meta, reader
        else:
            yield meta, _ArrowStream(reader)
    finally:
        reader.close()


class _ArrowStream:
    """Generic Arrow stream wrapper exposing the PyCapsule protocol."""

    def __init__(self, reader):
        self._reader = reader

    def __arrow_c_stream__(self, requested_schema=None):
        if requested_schema is not None:
            raise NotImplementedError("requested_schema is not supported")
        return self._reader.__arrow_c_stream__()


def _iter_aligned_batches(
    path_or_buffer,
    layer_info,
    attribute_fields,
    geometry_field,
    output_schema,
    open_kwargs,
    kwargs,
    max_features,
):
    """Yield batches of each layer, aligned to the common schema.

    Only one layer is open and being streamed at any moment.
    """
    for info in layer_info:
        with open_arrow(
            path_or_buffer, layer=info.name, return_fids=True, **open_kwargs, **kwargs
        ) as (_meta, reader):
            yield from _iter_layer_batches(
                reader,
                info,
                attribute_fields,
                geometry_field,
                output_schema,
                max_features,
            )


def _iter_layer_batches(
    reader, info, attribute_fields, geometry_field, output_schema, max_features
):
    emitted = 0
    for batch in reader:
        if max_features is not None:
            remaining = max_features - emitted
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)

        emitted += batch.num_rows
        yield _align_batch(
            batch, info, attribute_fields, geometry_field, output_schema
        )


def _align_batch(batch, info, attribute_fields, geometry_field, output_schema):
    """Align one source batch to the common output schema."""
    import pyarrow as pa

    number_of_rows = batch.num_rows
    source_names = batch.schema.names

    arrays = [
        pa.repeat(pa.scalar(info.name, type=pa.string()), number_of_rows),
        batch.column(info.fid_name).cast(pa.int64()),
    ]

    for field in attribute_fields:
        if field.name in source_names and field.name not in (
            info.geometry_name,
            info.fid_name,
        ):
            arrays.append(batch.column(field.name).cast(field.type))
        else:
            arrays.append(pa.nulls(number_of_rows, type=field.type))

    if geometry_field is not None:
        if info.geometry_name is not None:
            arrays.append(
                batch.column(info.geometry_name).cast(geometry_field.type)
            )
        else:
            arrays.append(pa.nulls(number_of_rows, type=geometry_field.type))

    return pa.RecordBatch.from_arrays(arrays, schema=output_schema)


class _LayerArrowInfo:
    """Arrow schema information for one layer."""

    def __init__(self, name, meta, schema):
        self.name = name
        self.meta = meta

        geometry_field = _arrow_geometry_field(schema, meta)
        self.geometry_name = (
            geometry_field.name if geometry_field is not None else None
        )
        self._geometry_type = (
            geometry_field.type if geometry_field is not None else None
        )
        self._geometry_metadata = (
            geometry_field.metadata if geometry_field is not None else None
        )

        fid_column = meta.get("fid_column")
        if fid_column and fid_column in schema.names:
            self.fid_name = fid_column
        else:
            raise ValueError(
                f"Layer {name!r} does not expose a feature id column; "
                "it cannot be combined with other layers"
            )

        self.attribute_fields = [
            field
            for field in schema
            if field.name not in (self.geometry_name, self.fid_name)
        ]


def _arrow_geometry_field(schema, meta):
    """Return the geometry field of an Arrow schema, or None."""
    for field in schema:
        extension_name = (field.metadata or {}).get(b"ARROW:extension:name")
        if extension_name in (b"geoarrow.wkb", b"ogc.wkb"):
            return field

    for name in (meta.get("geometry_name"), "wkb_geometry"):
        if name and name in schema.names:
            field = schema.field(name)
            if _is_binary_type(field.type):
                return field

    return None


def _is_binary_type(arrow_type):
    import pyarrow as pa

    return arrow_type == pa.binary() or arrow_type == pa.large_binary()


def _check_reserved_columns(attribute_fields, geometry_field):
    names = {field.name for field in attribute_fields}
    reserved = _RESERVED_COLUMNS & names
    if geometry_field is not None and GEOMETRY_COLUMN in names:
        reserved.add(GEOMETRY_COLUMN)
    if reserved:
        raise ValueError(
            "Source attribute names conflict with reserved output columns: "
            + ", ".join(sorted(reserved))
        )


def _crs_key(crs):
    """Return a normalized comparison key for a CRS value."""
    if crs is None:
        return None
    return str(crs).strip()


def _validate_layer_geometry(layer_info):
    """Validate geometry and CRS compatibility, returning the output geometry.

    Returns
    -------
    (geometry_field, geometry_type, crs)
        ``geometry_field`` is the unified Arrow field for the output (or None),
        ``geometry_type`` a name for the combined geometry type (or None) and
        ``crs`` the CRS shared by the spatial layers (or None).
    """
    import pyarrow as pa

    spatial = [info for info in layer_info if info.geometry_name is not None]

    if not spatial:
        return None, None, None

    crs_keys = {_crs_key(info.meta.get("crs")) for info in spatial}
    known_keys = crs_keys - {None}
    if len(known_keys) > 1 or (known_keys and None in crs_keys):
        detail = ", ".join(
            f"{info.name!r}: {info.meta.get('crs')!r}" for info in spatial
        )
        raise ValueError("Cannot combine layers with incompatible CRS: " + detail)

    crs = next(
        (info.meta.get("crs") for info in spatial if info.meta.get("crs") is not None),
        None,
    )

    geometry_types = {info.meta.get("geometry_type") for info in spatial}
    if len(geometry_types) == 1:
        geometry_type = next(iter(geometry_types))
    else:
        geometry_type = "Unknown"

    arrow_types = [info._geometry_type for info in spatial]
    try:
        unified = pa.unify_schemas(
            [pa.schema([pa.field(GEOMETRY_COLUMN, t)]) for t in arrow_types],
            promote_options="permissive",
        ).field(0)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
        raise ValueError(
            "Geometry columns of the selected layers cannot be aligned to a "
            f"common Arrow schema: {exc}"
        ) from exc

    metadata = next(
        (info._geometry_metadata for info in spatial if info._geometry_metadata),
        None,
    )
    geometry_field = pa.field(
        GEOMETRY_COLUMN, unified.type, nullable=True, metadata=metadata
    )
    return geometry_field, geometry_type, crs


def _unify_attribute_fields(layer_info):
    """Unify the attribute fields of all layers into one ordered list of fields."""
    import pyarrow as pa

    if not any(info.attribute_fields for info in layer_info):
        return []

    try:
        unified = pa.unify_schemas(
            [
                pa.schema([_nullable(field) for field in info.attribute_fields])
                for info in layer_info
            ],
            promote_options="permissive",
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
        detail = ", ".join(
            f"{info.name!r}: {[f.name for f in info.attribute_fields]}"
            for info in layer_info
        )
        raise ValueError(
            "Attribute fields of the selected layers cannot be aligned to a "
            f"common Arrow schema ({detail}): {exc}"
        ) from exc

    # Keep a stable column order: the order in which fields are first seen.
    ordered = []
    seen = set()
    for info in layer_info:
        for field in info.attribute_fields:
            if field.name in seen:
                continue
            seen.add(field.name)
            ordered.append(_nullable(unified.field(field.name)))
    return ordered


def _nullable(field):
    """Return a copy of an Arrow field marked as nullable.

    Fields that are absent from a layer are filled with nulls, so every aligned
    field must be nullable regardless of the source schema.
    """
    return field.with_nullable(True)


def _parse_options_names(xml):
    """Convert metadata xml to list of names."""
    # Based on Fiona's meta.py
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

    path, driver = _get_write_path_driver(path, driver, append=append)

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
