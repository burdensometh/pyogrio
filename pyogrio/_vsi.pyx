import fnmatch
import os
from io import BytesIO
from uuid import uuid4

from pyogrio._ogr cimport *
from pyogrio._ogr import _get_driver_metadata_item


cdef tuple get_ogr_vsimem_write_path(object path_or_fp, str driver):
    """Return the path to write to and whether it is a tmp vsimem filepath.

    If passed a io.BytesIO object to write to, a temporary vsimem file will be
    used to write the data directly to memory.
    Hence, a tuple will be returned with a /vsimem/ path and True to indicate
    the path will be to a tmp vsimem file.
    The path will have an extension inferred from the driver if possible. Path
    will be contained in an in-memory directory to contain sibling files
    (though drivers that create sibling files are not supported for in-memory
    files).

    Caller is responsible for deleting the directory via
    vsimem_rmtree_toplevel().

    Parameters
    ----------
    path_or_fp : str or io.BytesIO object
        Path or BytesIO object that you would like to write to.
    driver : str
        OGR driver name to be used for writing.

    Returns
    -------
    tuple of (path, use_tmp_vsimem)
        Tuple of the path to write to and a bool indicating if the path is a
        temporary vsimem filepath.

    """
    # The write path is not a BytesIO object, so return path as-is
    if not isinstance(path_or_fp, BytesIO):
        return (path_or_fp, False)

    # Check for existing bytes
    if path_or_fp.getbuffer().nbytes > 0:
        raise NotImplementedError(
            "writing to existing in-memory object is not supported"
        )

    # Create in-memory directory to contain auxiliary files.
    # Prefix with "pyogrio_" so it is clear the directory was created by pyogrio.
    memfilename = f"pyogrio_{uuid4().hex}"
    VSIMkdir(f"/vsimem/{memfilename}".encode("UTF-8"), 0666)

    # file extension is required for some drivers, set it based on driver metadata
    ext = ""
    recommended_ext = _get_driver_metadata_item(driver, "DMD_EXTENSIONS")
    if recommended_ext is not None:
        ext = "." + recommended_ext.split(" ")[0]

    path = f"/vsimem/{memfilename}/{memfilename}{ext}"

    return (path, True)


cdef str read_buffer_to_vsimem(bytes bytes_buffer):
    """Wrap the bytes (zero-copy) into an in-memory dataset.

    If the first 4 bytes indicate the bytes are a zip file, the returned path
    will be prefixed with /vsizip/ and suffixed with .zip to enable proper
    reading by GDAL.

    Caller is responsible for deleting the in-memory file via
    vsimem_rmtree_toplevel().

    Parameters
    ----------
    bytes_buffer : bytes
        Bytes to write to in-memory file

    Returns
    -------
    str
        Path to the in-memory file.

    """
    cdef int num_bytes = len(bytes_buffer)

    is_zipped = len(bytes_buffer) > 4 and bytes_buffer[:4].startswith(b"PK\x03\x04")
    ext = ".zip" if is_zipped else ""

    # Prefix with "pyogrio_" so it is clear the file was created by pyogrio.
    path = f"/vsimem/pyogrio_{uuid4().hex}{ext}"

    # Create an in-memory object that references bytes_buffer
    # NOTE: GDAL does not copy the contents of bytes_buffer; it must remain
    # in scope through the duration of using this file
    vsi_handle = VSIFileFromMemBuffer(
        path.encode("UTF-8"), <unsigned char *>bytes_buffer, num_bytes, 0
    )

    if vsi_handle == NULL:
        raise OSError("failed to read buffer into in-memory file")

    if VSIFCloseL(vsi_handle) != 0:
        raise OSError("failed to close in-memory file")

    if is_zipped:
        path = f"/vsizip/{path}"

    return path


cdef read_vsimem_to_buffer(str path, object out_buffer):
    """Copy bytes from in-memory file to buffer.

    This will automatically unlink the in-memory file pointed to by path; caller
    is still responsible for calling vsimem_rmtree_toplevel() to cleanup any
    other files contained in the in-memory directory.

    Parameters:
    -----------
    path : str
        path to in-memory file
    out_buffer : BytesIO object
        buffer to write bytes to

    """
    cdef unsigned char *vsi_buffer = NULL
    cdef vsi_l_offset vsi_buffer_size = 0

    try:
        # Take ownership of the buffer to avoid a copy; GDAL will automatically
        # unlink the memory file
        vsi_buffer = VSIGetMemFileBuffer(path.encode("UTF-8"), &vsi_buffer_size, 1)
        if vsi_buffer == NULL:
            raise RuntimeError("could not read bytes from in-memory file")

        # write bytes to buffer
        out_buffer.write(<bytes>vsi_buffer[:vsi_buffer_size])
        # rewind to beginning to allow caller to read
        out_buffer.seek(0)

    finally:
        if vsi_buffer != NULL:
            CPLFree(vsi_buffer)


cpdef str create_vsimem_directory():
    """Create a fresh empty in-memory directory.

    Caller is responsible for deleting the directory via
    vsimem_rmtree_toplevel().

    Returns
    -------
    str
        Path to the newly created in-memory directory.

    """
    cdef str directory = f"/vsimem/pyogrio_{uuid4().hex}"

    if VSIMkdir(directory.encode("UTF-8"), 0666) != 0:
        raise OSError(f"failed to create in-memory directory '{directory}'")

    return directory


cpdef str write_files_to_vsimem(dict files):
    """Write an explicit group of files into a fresh in-memory directory.

    Each member is stored under its own name in a newly created
    ``/vsimem/pyogrio_<uuid>/`` directory. The bytes buffers are wrapped
    zero-copy; they must remain referenced by the caller for as long as the
    in-memory dataset is in use.

    All input is validated before anything is created, so that invalid input
    never leaves partial in-memory data behind.

    Caller is responsible for deleting the directory via
    vsimem_rmtree_toplevel().

    Parameters
    ----------
    files : dict
        Mapping of member file name (``str`` or ``path-like``) to bytes-like
        buffer. Names must be plain file names without directory components.

    Returns
    -------
    str
        Path to the in-memory directory containing the files.

    """
    cdef bytes bytes_buffer
    cdef int num_bytes
    cdef VSILFILE* vsi_handle = NULL
    cdef str directory
    cdef str member
    cdef str member_path

    if not files:
        raise ValueError("at least one member file is required")

    # Validate everything up front so that failure does not leave partial
    # in-memory data behind.
    normalized = {}
    for name, buffer in files.items():
        if not isinstance(name, str):
            name = os.fspath(name)

        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise ValueError(
                f"member names must be plain file names without directory "
                f"components: got '{name}'"
            )
        member = name

        if isinstance(buffer, bytearray):
            # VSIFileFromMemBuffer requires a contiguous bytes-like object;
            # copy bytearray into an immutable bytes object.
            buffer = bytes(buffer)
        elif not isinstance(buffer, bytes):
            raise TypeError(
                f"buffer for member '{name}' must be bytes or bytearray, "
                f"got {type(buffer).__name__}"
            )

        if member in normalized:
            raise ValueError(f"duplicate member file name: '{member}'")

        normalized[member] = buffer

    directory = create_vsimem_directory()

    try:
        for member, bytes_buffer in normalized.items():
            num_bytes = len(bytes_buffer)
            member_path = f"{directory}/{member}"
            # NOTE: GDAL does not copy the contents of bytes_buffer; the dict
            # (and thus the bytes objects) must stay in scope while the file
            # is in use.
            vsi_handle = VSIFileFromMemBuffer(
                member_path.encode("UTF-8"),
                <unsigned char *>bytes_buffer,
                num_bytes,
                0,
            )
            if vsi_handle == NULL:
                raise OSError(
                    f"failed to write member '{member}' into in-memory directory"
                )

            if VSIFCloseL(vsi_handle) != 0:
                raise OSError(
                    f"failed to close in-memory file for member '{member}'"
                )
    except BaseException:
        # never deliver a partially populated in-memory directory
        vsimem_rmtree_toplevel(directory)
        raise

    return directory


cdef bytes read_vsimem_file(str path):
    """Read the full contents of an in-memory file into bytes.

    The in-memory file itself is left untouched; the caller remains
    responsible for cleaning up the in-memory directory.

    Parameters
    ----------
    path : str
        Path to the in-memory file.

    Returns
    -------
    bytes
        Copy of the file contents.

    """
    cdef unsigned char *vsi_buffer = NULL
    cdef vsi_l_offset vsi_buffer_size = 0
    cdef bytes result

    # take_ownership=0: inspect the buffer without unlinking the file and
    # without taking ownership; the returned pointer must NOT be freed here
    vsi_buffer = VSIGetMemFileBuffer(path.encode("UTF-8"), &vsi_buffer_size, 0)
    if vsi_buffer == NULL:
        raise RuntimeError(f"could not read bytes from in-memory file '{path}'")

    # slicing copies the contents into a fresh bytes object, so the result
    # does not depend on GDAL's in-memory buffer
    result = <bytes>vsi_buffer[:vsi_buffer_size]

    return result


cpdef dict read_vsimem_files(str directory):
    """Read every file directly contained in an in-memory directory.

    Only regular files at the top level of the directory are returned;
    subdirectories are skipped.

    Parameters
    ----------
    directory : str
        Path to an existing in-memory directory.

    Returns
    -------
    dict
        Mapping of member file name to a bytes copy of its contents.

    """
    cdef const char *directory_c
    cdef int n
    cdef char** papszFiles
    cdef VSIStatBufL st_buf
    cdef str member
    cdef str member_path

    directory_b = directory.encode("UTF-8")
    directory_c = directory_b

    if not VSIStatL(directory_c, &st_buf) == 0:
        raise FileNotFoundError(f"Path does not exist: '{directory}'")
    if not VSI_ISDIR(st_buf.st_mode):
        raise NotADirectoryError(f"Path is not a directory: '{directory}'")

    try:
        papszFiles = VSIReadDirRecursive(directory_c)
        n = CSLCount(<CSLConstList>papszFiles)
        members = []
        for i in range(n):
            member = papszFiles[i].decode("UTF-8")
            # only read top-level regular files; skip nested directories
            if "/" in member:
                continue
            members.append(member)
    finally:
        CSLDestroy(papszFiles)

    if not directory.endswith("/"):
        directory = f"{directory}/"

    return {member: read_vsimem_file(f"{directory}{member}") for member in members}


cpdef vsimem_rmtree_toplevel(str path):
    """Remove the top-level file or top-level directory containing the file.

    This is used for final cleanup of an in-memory dataset. The path can point
    to either:
    - a top-level file (directly in /vsimem/).
    - a file in a directory, which may include sibling files.
    - a zip file (reported as a directory by VSI_ISDIR).

    Except for the first case, the top-level directory (direct subdirectory of
    /vsimem/) will be determined and will be removed recursively.

    Additional VSI handlers may be chained to the left of /vsimem/ in path and
    will be ignored.

    Even though it is only meant for "internal use", the function is declared
    as cpdef, so it can be called from tests as well.

    Parameters:
    -----------
    path : str
        path to in-memory file

    """
    cdef VSIStatBufL st_buf

    if "/vsimem/" not in path:
        raise ValueError(f"Path is not a /vsimem/ path: '{path}'")

    # Determine the top-level directory of the file
    mempath_parts = path.split("/vsimem/")[1].split("/")
    if len(mempath_parts) == 0:
        raise OSError("path to in-memory file or directory is required")

    toplevel_path = f"/vsimem/{mempath_parts[0]}"

    if not VSIStatL(toplevel_path.encode("UTF-8"), &st_buf) == 0:
        raise FileNotFoundError(f"Path does not exist: '{path}'")

    if VSI_ISDIR(st_buf.st_mode):
        errcode = VSIRmdirRecursive(toplevel_path.encode("UTF-8"))
    else:
        errcode = VSIUnlink(toplevel_path.encode("UTF-8"))

    if errcode != 0:
        raise OSError(f"Error removing '{path}': {errcode=}")


def ogr_vsi_listtree(str path, str pattern):
    """Recursively list the contents in a VSI directory.

    An fnmatch pattern can be specified to filter the directories/files
    returned.

    Parameters:
    -----------
    path : str
        Path to the VSI directory to be listed.
    pattern : str
        Pattern to filter results, in fnmatch format.

    """
    cdef const char *path_c
    cdef int n
    cdef char** papszFiles
    cdef VSIStatBufL st_buf

    path_b = path.encode("UTF-8")
    path_c = path_b

    if not VSIStatL(path_c, &st_buf) == 0:
        raise FileNotFoundError(f"Path does not exist: '{path}'")
    if not VSI_ISDIR(st_buf.st_mode):
        raise NotADirectoryError(f"Path is not a directory: '{path}'")

    try:
        papszFiles = VSIReadDirRecursive(path_c)
        n = CSLCount(<CSLConstList>papszFiles)
        files = []
        for i in range(n):
            files.append(papszFiles[i].decode("UTF-8"))
    finally:
        CSLDestroy(papszFiles)

    # Apply filter pattern
    if pattern is not None:
        files = fnmatch.filter(files, pattern)

    # Prepend files with the base path
    if not path.endswith("/"):
        path = f"{path}/"
    files = [f"{path}{file}" for file in files]

    return files


def ogr_vsi_rmtree(str path):
    """Recursively remove VSI directory.

    Parameters:
    -----------
    path : str
        path to the VSI directory to be removed.

    """
    cdef const char *path_c
    cdef VSIStatBufL st_buf

    try:
        path_b = path.encode("UTF-8")
    except UnicodeDecodeError:
        path_b = path
    path_c = path_b
    if not VSIStatL(path_c, &st_buf) == 0:
        raise FileNotFoundError(f"Path does not exist: '{path}'")
    if not VSI_ISDIR(st_buf.st_mode):
        raise NotADirectoryError(f"Path is not a directory: '{path}'")
    if path.endswith("/vsimem") or path.endswith("/vsimem/"):
        raise OSError("path to in-memory file or directory is required")

    errcode = VSIRmdirRecursive(path_c)
    if errcode != 0:
        raise OSError(f"Error in rmtree of '{path}': {errcode=}")


def ogr_vsi_unlink(str path):
    """Remove VSI file.

    Parameters:
    -----------
    path : str
        path to the VSI file to be removed.

    """
    cdef const char *path_c
    cdef VSIStatBufL st_buf

    try:
        path_b = path.encode("UTF-8")
    except UnicodeDecodeError:
        path_b = path
    path_c = path_b

    if not VSIStatL(path_c, &st_buf) == 0:
        raise FileNotFoundError(f"Path does not exist: '{path}'")

    if VSI_ISDIR(st_buf.st_mode):
        raise IsADirectoryError(f"Path is a directory: '{path}'")

    errcode = VSIUnlink(path_c)
    if errcode != 0:
        raise OSError(f"Error removing '{path}': {errcode=}")


def ogr_vsi_curl_clear_all_cache():
    """Clean local cache associated with /vsicurl/.

    """

    VSICurlClearCache()


def ogr_vsi_curl_clear_cache(str prefix):
    """Clean local cache associated with /vsicurl/.

    Parameters
    ----------
    prefix : str
        Filename or prefix to clear associated cache.

    """
    cdef const char *prefix_c

    try:
        prefix_b = prefix.encode("UTF-8")
    except UnicodeDecodeError:
        prefix_b = prefix
    prefix_c = prefix_b

    VSICurlPartialClearCache(prefix_c)
