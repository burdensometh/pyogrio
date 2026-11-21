"""Utility functions."""

import contextlib
import os
import re
import sys
import uuid
import warnings
from packaging.version import Version
from pathlib import Path
from urllib.parse import urlparse

from pyogrio._ogr import MULTI_EXTENSIONS
from pyogrio._vsi import vsimem_rmtree_toplevel as _vsimem_rmtree_toplevel
from pyogrio.errors import DataSourceError


def get_vsi_path_or_buffer(path_or_buffer):
    """Get VSI-prefixed path or bytes buffer depending on type of path_or_buffer.

    If path_or_buffer is a bytes object, it will be returned directly and will
    be read into an in-memory dataset when passed to one of the Cython functions.

    If path_or_buffer is a file-like object with a read method, bytes will be
    read from the file-like object and returned.

    Otherwise, it will be converted to a string, and parsed to prefix with
    appropriate GDAL /vsi*/ prefixes.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path, bytes, or file-like
        A dataset path or URI, raw buffer, or file-like object with a read method.

    Returns
    -------
    str or bytes

    """
    # treat Path objects here already to ignore their read method + to avoid backslashes
    # on Windows.
    if isinstance(path_or_buffer, Path):
        return vsi_path(path_or_buffer)

    if isinstance(path_or_buffer, bytes):
        return path_or_buffer

    if hasattr(path_or_buffer, "read"):
        bytes_buffer = path_or_buffer.read()

        # rewind buffer if possible so that subsequent operations do not need to rewind
        if hasattr(path_or_buffer, "seekable") and path_or_buffer.seekable():
            path_or_buffer.seek(0)

        return bytes_buffer

    return vsi_path(str(path_or_buffer))


def vsi_path(path: str | Path) -> str:
    """Ensure path is a local path or a GDAL-compatible VSI path."""
    # Convert Path objects to string, but for VSI paths, keep posix style path.
    if isinstance(path, Path):
        if sys.platform == "win32" and path.as_posix().startswith("/vsi"):
            path = path.as_posix()
        else:
            path = str(path)

    # path is already in GDAL format
    if path.startswith("/vsi"):
        return path

    # Windows drive letters (e.g. "C:\") confuse `urlparse` as they look like
    # URL schemes
    if sys.platform == "win32" and re.match("^[a-zA-Z]\\:", path):
        # If it is not a zip file or it is multi-extension zip file that is directly
        # supported by a GDAL driver, return the path as is.
        if not path.split("!")[0].endswith(".zip"):
            return path
        if path.split("!")[0].endswith(MULTI_EXTENSIONS):
            return path

        # prefix then allow to proceed with remaining parsing
        path = f"zip://{path}"

    path, archive, scheme = _parse_uri(path)

    if (
        scheme
        or archive
        or (path.endswith(".zip") and not path.endswith(MULTI_EXTENSIONS))
    ):
        return _construct_vsi_path(path, archive, scheme)

    return path


# Supported URI schemes and their mapping to GDAL's VSI suffix.
SCHEMES = {
    "file": "file",
    "zip": "zip",
    "tar": "tar",
    "gzip": "gzip",
    "http": "curl",
    "https": "curl",
    "ftp": "curl",
    "s3": "s3",
    "gs": "gs",
    "az": "az",
    "adls": "adls",
    "adl": "adls",  # fsspec uses this
    "hdfs": "hdfs",
    "webhdfs": "webhdfs",
    # GDAL additionally supports oss and swift for remote filesystems, but
    # those are for now not added as supported URI
}

CURLSCHEMES = {k for k, v in SCHEMES.items() if v == "curl"}


def _parse_uri(path: str):
    """Parse a URI.

    Returns a tuples of (path, archive, scheme)

    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    parts = urlparse(path, allow_fragments=False)

    # if the scheme is not one of GDAL's supported schemes, return raw path
    if parts.scheme and not all(p in SCHEMES for p in parts.scheme.split("+")):
        return path, "", ""

    # we have a URI
    path = parts.path
    scheme = parts.scheme or ""

    if parts.query:
        path += "?" + parts.query

    if parts.scheme and parts.netloc:
        path = parts.netloc + path

    parts = path.split("!")
    path = parts.pop() if parts else ""
    archive = parts.pop() if parts else ""
    return (path, archive, scheme)


def _construct_vsi_path(path, archive, scheme) -> str:
    """Convert a parsed path to a GDAL VSI path."""
    prefix = ""
    suffix = ""
    schemes = scheme.split("+")

    if "zip" not in schemes and (
        archive.endswith(".zip")
        or (path.endswith(".zip") and not path.endswith(MULTI_EXTENSIONS))
    ):
        schemes.insert(0, "zip")

    if schemes:
        prefix = "/".join(f"vsi{SCHEMES[p]}" for p in schemes if p and p != "file")

        if schemes[-1] in CURLSCHEMES:
            suffix = f"{schemes[-1]}://"

    if prefix:
        if archive:
            return "/{}/{}{}/{}".format(prefix, suffix, archive, path.lstrip("/"))
        else:
            return f"/{prefix}/{suffix}{path}"

    return path


def _preprocess_options_key_value(options):
    """Preprocess options.

    For example, `spatial_index=True` gets converted to `SPATIAL_INDEX="YES"`.
    """
    if not isinstance(options, dict):
        raise TypeError(f"Expected options to be a dict, got {type(options)}")

    result = {}
    for k, v in options.items():
        if v is None:
            continue
        k = k.upper()
        if isinstance(v, bool):
            v = "ON" if v else "OFF"
        else:
            v = str(v)
        result[k] = v
    return result


def _mask_to_wkb(mask):
    """Convert a Shapely mask geometry to WKB.

    Parameters
    ----------
    mask : Shapely geometry
        The geometry to convert to WKB.

    Returns
    -------
    WKB bytes or None

    Raises
    ------
    ValueError
        raised if Shapely >= 2.0 is not available or mask is not a Shapely
        Geometry object

    """
    if mask is None:
        return mask

    try:
        import shapely

        if Version(shapely.__version__) < Version("2.0.0"):
            shapely = None
    except ImportError:
        shapely = None

    if not shapely:
        raise ValueError("'mask' parameter requires Shapely >= 2.0")

    if not isinstance(mask, shapely.Geometry):
        raise ValueError("'mask' parameter must be a Shapely geometry")

    return shapely.to_wkb(mask)


def vsimem_rmtree_toplevel(path: str | Path):
    """Remove the parent directory of the file path recursively.

    This is used for final cleanup of an in-memory dataset, which may have been
    created within a directory to contain sibling files.

    Additional VSI handlers may be chained to the left of /vsimem/ in path and
    will be ignored.

    Remark: function is defined here to be able to run tests on it.

    Parameters
    ----------
    path : str or pathlib.Path
        path to in-memory file

    """
    if isinstance(path, Path):
        path = path.as_posix()

    _vsimem_rmtree_toplevel(path)


# Drivers for which atomic publishing of a local file is supported.  This is
# limited to formats that consist of a single file; formats that are made up of
# multiple files (e.g. ESRI Shapefile with its .shp/.dbf/.prj/... sidecar
# files) cannot be replaced atomically as a single unit.
_ATOMIC_WRITE_SUPPORTED_DRIVERS = {"GPKG"}

# Sidecar files that SQLite-based formats (e.g. GPKG) may create while writing
_SQLITE_SIDECAR_SUFFIXES = ("-journal", "-shm", "-wal")


def _validate_atomic_write(path, driver, append):
    """Validate that atomic publishing is supported for the given output.

    Raises
    ------
    ValueError
        If atomic publishing is not supported for the given path, driver or
        append mode.

    """
    if append:
        raise ValueError("'atomic' is not supported in combination with append")

    if not isinstance(path, str) or path.startswith("/vsi"):
        raise ValueError(
            "'atomic' is only supported for writing to a local file path; it is "
            "not supported for in-memory files or /vsi/ paths"
        )

    if driver not in _ATOMIC_WRITE_SUPPORTED_DRIVERS:
        supported = ", ".join(sorted(_ATOMIC_WRITE_SUPPORTED_DRIVERS))
        raise ValueError(
            f"'atomic' is not supported for the {driver!r} driver (currently only "
            f"supported for: {supported}). Formats that consist of multiple files "
            "(e.g. ESRI Shapefile with its sidecar files) cannot be replaced "
            "atomically as a single file."
        )


def _default_layer_name(path):
    """Get the default layer name for a new layer written to path.

    Mirrors the logic used when writing (based on the file name), so that a
    layer can be created with the correct name while writing to a staging file
    with a different file name.
    """
    # For multi extensions (e.g. ".gpkg.zip"), strip the full extension
    for multi_ext in MULTI_EXTENSIONS:
        if path.endswith(multi_ext):
            return os.path.split(path)[1][: -len(multi_ext)]

    return os.path.splitext(os.path.split(path)[1])[0]


def _verify_staging_file(staging_path, layer):
    """Verify a fully written and closed staging file before publishing it.

    Opens the file with a fresh dataset handle and checks that the expected
    layer is present.  Raises DataSourceError if the file is not valid.
    """
    # local import: requires the GDAL environment to be initialized, which is
    # only guaranteed once pyogrio.core has been imported
    from pyogrio._io import ogr_list_layers

    try:
        layers = ogr_list_layers(str(staging_path))
    except Exception as exc:
        raise DataSourceError(
            "atomic write verification failed: could not open the written file "
            f"{str(staging_path)!r}: {exc}"
        ) from exc

    layer_names = [layer_info[0] for layer_info in layers]
    if not len(layer_names):
        raise DataSourceError(
            "atomic write verification failed: the written file "
            f"{str(staging_path)!r} contains no layers"
        )

    if layer is not None and layer not in layer_names:
        raise DataSourceError(
            f"atomic write verification failed: layer {layer!r} not found in "
            f"the written file {str(staging_path)!r}"
        )


def _remove_staging_files(staging_path):
    """Remove a staging file and any of its sidecar files.

    Only ever removes files derived from the staging path, never the target
    file of an atomic publish.
    """
    for suffix in ("", *_SQLITE_SIDECAR_SUFFIXES):
        staging_file = str(staging_path) + suffix
        try:
            os.remove(staging_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            warnings.warn(
                f"could not remove staging file {staging_file!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


@contextlib.contextmanager
def _atomic_publish(path, layer=None):
    """Context manager to publish a local file atomically.

    Yields a staging path in the same directory as ``path`` to write the data
    to.  When the enclosed write completes without error, the staging file is
    verified (see :func:`_verify_staging_file`) and then moved over ``path``
    with ``os.replace`` (atomic because the staging file is on the same file
    system).  If the write fails or is interrupted, the staging file is
    removed and a previously existing target file is left untouched.

    Once the staging file has been moved over the target, the result is
    considered committed and is never cleaned up as a staging file.

    Parameters
    ----------
    path : str
        target file path (local file system)
    layer : str, optional
        name of the layer that should be present in the written file; used to
        verify the staging file before publishing

    Yields
    ------
    str
        path of the staging file to write to

    """
    target = Path(path)
    staging_path = target.with_name(
        f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp{target.suffix}"
    )
    committed = False
    try:
        yield str(staging_path)

        # the dataset has been closed by the write; verify before publishing
        _verify_staging_file(staging_path, layer)

        os.replace(staging_path, target)
        committed = True

    finally:
        if not committed:
            _remove_staging_files(staging_path)
