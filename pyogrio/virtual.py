"""In-memory multi-file vector datasets.

Some vector formats (e.g. ESRI Shapefile) are spread over several sibling
files with distinct semantics (``.shp``, ``.shx``, ``.dbf``, ``.prj``,
``.cpg`` ...). The functions in this module allow such a dataset to be
provided as an explicit mapping of member file name to bytes buffer instead
of writing the members to a temporary directory and reconstructing their
relationship from file names.

All members are staged together in a GDAL ``/vsimem/`` directory, which is
removed again when the reader or writer is closed.
"""

import os
from uuid import uuid4

from pyogrio._env import GDALEnv

with GDALEnv():
    from pyogrio._ogr import ogr_driver_supports_vsi
    from pyogrio._vsi import (
        create_vsimem_directory,
        read_vsimem_files,
        vsimem_rmtree_toplevel as _vsimem_rmtree_toplevel,
        write_files_to_vsimem,
    )


# Multi-file drivers and the member extensions they require. The required
# members must all be present up front so that failure happens before GDAL is
# asked to open an incomplete dataset. Optional members (e.g. ``.prj`` with
# the CRS or ``.cpg`` with the character encoding) are passed through
# untouched when present.
MULTI_FILE_DRIVERS = {
    "ESRI Shapefile": {
        "required_extensions": (".shp", ".shx", ".dbf"),
        "main_extension": ".shp",
    },
}


def _normalize_member_name(name):
    """Validate a member mapping key and return it as a plain file name."""
    if not isinstance(name, str):
        name = os.fspath(name)

    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError(
            "member names must be plain file names without directory "
            f"components: got '{name}'"
        )

    return name


def _coerce_buffer(name, buffer):
    """Turn a bytes-like or readable file-like member into bytes."""
    if isinstance(buffer, bytearray):
        return bytes(buffer)

    if isinstance(buffer, bytes):
        return buffer

    if hasattr(buffer, "read"):
        data = buffer.read()
        # rewind so the caller can reuse the object
        if hasattr(buffer, "seekable") and buffer.seekable():
            buffer.seek(0)
        if not isinstance(data, bytes):
            raise TypeError(
                f"file-like member '{name}' must return bytes from read(), "
                f"got {type(data).__name__}"
            )
        return data

    raise TypeError(
        f"buffer for member '{name}' must be bytes, bytearray or a readable "
        f"file-like object, got {type(buffer).__name__}"
    )


def _prepare_members(files):
    """Validate and normalize the explicit member mapping."""
    if not isinstance(files, dict):
        raise TypeError(
            "members must be provided as a dict of file name to buffer, "
            f"got {type(files).__name__}"
        )
    if not files:
        raise ValueError("at least one member file is required")

    members = {}
    for name, buffer in files.items():
        member = _normalize_member_name(name)
        if member in members:
            raise ValueError(f"duplicate member file name: '{member}'")
        members[member] = _coerce_buffer(member, buffer)

    return members


def _detect_multi_file_driver(members):
    """Detect the driver from member extensions and validate requirements.

    Returns a tuple of ``(driver name, main file name)``. Fails early when a
    required member is missing.
    """
    # map lower-cased extension back to the member name with original casing
    extension_to_name = {
        os.path.splitext(name)[1].lower(): name for name in members
    }
    extensions = set(extension_to_name)

    for driver, spec in MULTI_FILE_DRIVERS.items():
        if spec["main_extension"] not in extensions:
            continue

        missing = [
            ext for ext in spec["required_extensions"] if ext not in extensions
        ]
        if missing:
            raise ValueError(
                f"'{driver}' dataset is missing required member(s): "
                f"{', '.join(missing)}"
            )

        return driver, extension_to_name[spec["main_extension"]]

    raise ValueError(
        "could not determine a multi-file vector driver from the member "
        f"extensions: {sorted(extensions)}"
    )


class VirtualVectorReader:
    """Read a multi-file vector dataset provided as an explicit member list.

    The members are staged in a private GDAL ``/vsimem/`` directory and the
    virtual path of the main file can be handed to any of the regular pyogrio
    read entry points (``read_dataframe``, ``read``, ``read_info``,
    ``list_layers``, ``open_arrow`` ...). The in-memory data is removed when
    the reader is closed, including when entering the context fails.

    Character encoding and CRS are not reinterpreted: optional members such as
    ``.cpg`` (encoding) and ``.prj`` (CRS) are staged exactly as provided, so
    GDAL applies their semantics the same way as for files on disk.

    Parameters
    ----------
    files : dict
        Explicit mapping of member file name (``str``) to bytes buffer,
        bytearray, or readable file-like object. Member names must be plain
        file names without directory components.
    layer : str or int, optional
        Layer to select from the virtual dataset. Required when the dataset
        contains more than one layer; defaults to the single layer otherwise.

    Attributes
    ----------
    path : str
        GDAL ``/vsimem/`` path of the main member file.
    driver : str
        OGR driver name detected from the members.
    layer : str
        Resolved layer name to read.

    Examples
    --------
    >>> members = {
    ...     "test.shp": shp_bytes,
    ...     "test.shx": shx_bytes,
    ...     "test.dbf": dbf_bytes,
    ...     "test.prj": prj_bytes,
    ... }
    >>> with open_virtual_vector_files(members) as source:
    ...     df = pyogrio.read_dataframe(source.path, layer=source.layer)

    """

    def __init__(self, files, layer=None):
        self._files = files
        self._requested_layer = layer
        self._directory = None
        self._driver = None
        self._main_file = None
        self._path = None
        self._layer = None
        self._closed = False
        # keep references to the staged bytes; GDAL wraps them zero-copy and
        # they must stay alive while the dataset is open
        self._members = None

    def __enter__(self):
        try:
            self._members = _prepare_members(self._files)
            self._driver, self._main_file = _detect_multi_file_driver(
                self._members
            )
            self._directory = write_files_to_vsimem(self._members)
            self._path = f"{self._directory}/{self._main_file}"

            # resolve the layer up front so that ambiguous datasets fail here
            # instead of silently defaulting to the first layer
            from pyogrio.core import list_layers

            layer_names = [row[0] for row in list_layers(self._path)]

            if self._requested_layer is None:
                if len(layer_names) > 1:
                    raise ValueError(
                        f"dataset contains multiple layers {layer_names}; "
                        "the 'layer' parameter must be specified"
                    )
                self._layer = layer_names[0]
            elif isinstance(self._requested_layer, int):
                index = self._requested_layer
                if not 0 <= index < len(layer_names):
                    raise ValueError(
                        f"layer index {index} is out of range for "
                        f"{len(layer_names)} layer(s)"
                    )
                self._layer = layer_names[index]
            else:
                if self._requested_layer not in layer_names:
                    raise ValueError(
                        f"layer '{self._requested_layer}' not found in "
                        f"dataset; available layers: {layer_names}"
                    )
                self._layer = self._requested_layer

            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Remove the in-memory dataset. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._cleanup()
        self._members = None

    def _cleanup(self):
        if self._directory is not None:
            try:
                _vsimem_rmtree_toplevel(self._directory)
            except FileNotFoundError:
                pass
            self._directory = None

    def _check_open(self):
        if self._closed:
            raise ValueError("VirtualVectorReader is already closed")
        if self._directory is None:
            raise ValueError(
                "VirtualVectorReader must be opened as a context manager "
                "before use"
            )

    @property
    def path(self):
        """GDAL /vsimem/ path of the main member file."""
        self._check_open()
        return self._path

    @property
    def driver(self):
        """OGR driver name detected from the members."""
        self._check_open()
        return self._driver

    @property
    def layer(self):
        """Resolved layer name to read."""
        self._check_open()
        return self._layer

    def __repr__(self):
        if self._closed:
            state = "closed"
        elif self._directory is None:
            state = "unopened"
        else:
            state = f"driver={self._driver!r}"
        return f"<VirtualVectorReader({state})>"


class VirtualVectorWriter:
    """Write a multi-file vector dataset entirely in memory.

    A private GDAL ``/vsimem/`` directory is created on entry and its main
    file path is used as the write target. On successful close, every member
    produced by the driver is read back into a mapping of member name to
    bytes, which can be reopened directly with
    :func:`open_virtual_vector_files`. If writing fails, the partial
    in-memory data is discarded and no incomplete member set is delivered.

    Parameters
    ----------
    driver : str
        OGR driver name of a multi-file driver (e.g.
        ``"ESRI Shapefile"``).
    layer : str, optional
        Layer name to create. A unique name is generated when omitted.

    Attributes
    ----------
    path : str
        GDAL ``/vsimem/`` path of the main member file to write to.
    driver : str
        OGR driver name.
    layer : str
        Layer name to create.

    Examples
    --------
    >>> with open_virtual_vector_files_for_write(
    ...     "ESRI Shapefile", layer="test"
    ... ) as target:
    ...     pyogrio.write_dataframe(
    ...         df, target.path, layer=target.layer, driver=target.driver
    ...     )
    ...     members = target.close_and_get_files()

    """

    def __init__(self, driver, layer=None):
        self._driver = driver
        self._requested_layer = layer
        self._directory = None
        self._layer = None
        self._main_file = None
        self._path = None
        self._closed = False

    def __enter__(self):
        try:
            if self._driver not in MULTI_FILE_DRIVERS:
                raise ValueError(
                    f"driver '{self._driver}' does not produce a multi-file "
                    "dataset; supported drivers: "
                    f"{sorted(MULTI_FILE_DRIVERS)}"
                )

            if not ogr_driver_supports_vsi(self._driver):
                raise ValueError(
                    f"driver '{self._driver}' does not support in-memory "
                    "(/vsimem/) I/O in this GDAL build"
                )

            spec = MULTI_FILE_DRIVERS[self._driver]
            base_name = f"pyogrio_{uuid4().hex}"
            self._directory = create_vsimem_directory()
            self._main_file = f"{base_name}{spec['main_extension']}"
            self._path = f"{self._directory}/{self._main_file}"
            self._layer = self._requested_layer or base_name

            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Discard the in-memory dataset without returning members."""
        if self._closed:
            return
        self._closed = True
        self._cleanup()

    def close_and_get_files(self):
        """Read back all members, then remove the in-memory dataset.

        Returns
        -------
        dict
            Mapping of member file name to bytes. Raises if the required
            members were not produced; in that case the partial data is
            cleaned up and nothing is delivered.
        """
        self._check_open()

        try:
            members = read_vsimem_files(self._directory)

            spec = MULTI_FILE_DRIVERS[self._driver]
            present_extensions = {
                os.path.splitext(name)[1].lower() for name in members
            }
            missing = [
                ext
                for ext in spec["required_extensions"]
                if ext not in present_extensions
            ]
            if missing:
                raise OSError(
                    f"driver '{self._driver}' did not produce required "
                    f"member(s): {', '.join(missing)}"
                )

            return members
        finally:
            self.close()

    def _cleanup(self):
        if self._directory is not None:
            try:
                _vsimem_rmtree_toplevel(self._directory)
            except FileNotFoundError:
                pass
            self._directory = None

    def _check_open(self):
        if self._closed:
            raise ValueError("VirtualVectorWriter is already closed")
        if self._directory is None:
            raise ValueError(
                "VirtualVectorWriter must be opened as a context manager "
                "before use"
            )

    @property
    def path(self):
        """GDAL /vsimem/ path of the main member file to write to."""
        self._check_open()
        return self._path

    @property
    def driver(self):
        """OGR driver name."""
        return self._driver

    @property
    def layer(self):
        """Layer name to create."""
        self._check_open()
        return self._layer

    def __repr__(self):
        if self._closed:
            state = "closed"
        elif self._directory is None:
            state = f"unopened driver={self._driver!r}"
        else:
            state = f"driver={self._driver!r}"
        return f"<VirtualVectorWriter({state})>"


def open_virtual_vector_files(files, layer=None):
    """Open a multi-file vector dataset from an explicit member list.

    Parameters
    ----------
    files : dict
        Explicit mapping of member file name to bytes buffer, bytearray, or
        readable file-like object. Member names must be plain file names
        without directory components. Required members of the detected driver
        (e.g. ``.shp``, ``.shx``, ``.dbf`` for ESRI Shapefile) must all be
        present. Optional members such as ``.prj`` (CRS) and ``.cpg``
        (encoding) are staged as provided and interpreted by GDAL.
    layer : str or int, optional
        Layer to select. Required when the dataset contains multiple layers.

    Returns
    -------
    VirtualVectorReader
        Context manager exposing the virtual ``path`` and resolved ``layer``.
        The in-memory data is removed when the reader is closed.

    """
    return VirtualVectorReader(files, layer=layer)


def open_virtual_vector_files_for_write(driver, layer=None):
    """Create an in-memory target for a multi-file vector dataset.

    Parameters
    ----------
    driver : str
        OGR driver name of a multi-file driver (e.g.
        ``"ESRI Shapefile"``).
    layer : str, optional
        Layer name to create. A unique name is generated when omitted.

    Returns
    -------
    VirtualVectorWriter
        Context manager exposing the virtual ``path`` and ``layer`` to write
        to. Use :meth:`VirtualVectorWriter.close_and_get_files` to obtain the
        resulting members as a mapping of file name to bytes.

    """
    return VirtualVectorWriter(driver, layer=layer)
