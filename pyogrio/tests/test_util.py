from pathlib import Path

from pyogrio import vsi_listtree, vsi_unlink
from pyogrio.errors import DataSourceError
from pyogrio.raw import read, write
from pyogrio.util import (
    _atomic_publish,
    _default_layer_name,
    _validate_atomic_write,
    vsimem_rmtree_toplevel,
)

import pytest


def test_vsimem_rmtree_toplevel(naturalearth_lowres):
    # Prepare test data in /vsimem/
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["spatial_index"] = False
    meta["geometry_type"] = "MultiPolygon"
    test_dir_path = Path(f"/vsimem/test/{naturalearth_lowres.stem}.gpkg")
    test_dir2_path = Path(f"/vsimem/test2/test2/{naturalearth_lowres.stem}.gpkg")

    write(test_dir_path, geometry, field_data, **meta)
    write(test_dir2_path, geometry, field_data, **meta)

    # Check if everything was created properly with listtree
    files = vsi_listtree("/vsimem/")
    assert test_dir_path.as_posix() in files
    assert test_dir2_path.as_posix() in files

    # Test deleting parent dir of file in single directory
    vsimem_rmtree_toplevel(test_dir_path)
    files = vsi_listtree("/vsimem/")
    assert test_dir_path.parent.as_posix() not in files
    assert test_dir2_path.as_posix() in files

    # Test deleting top-level dir of file in a subdirectory
    vsimem_rmtree_toplevel(test_dir2_path)
    assert test_dir2_path.as_posix() not in vsi_listtree("/vsimem/")


def test_vsimem_rmtree_toplevel_error(naturalearth_lowres):
    # Prepare test data in /vsimem
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["spatial_index"] = False
    meta["geometry_type"] = "MultiPolygon"
    test_file_path = Path(f"/vsimem/pyogrio_test_{naturalearth_lowres.stem}.gpkg")

    write(test_file_path, geometry, field_data, **meta)
    assert test_file_path.as_posix() in vsi_listtree("/vsimem/")

    # Deleting parent dir of non-existent file should raise an error.
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        vsimem_rmtree_toplevel("/vsimem/test/non-existent.gpkg")

    # File should still be there
    assert test_file_path.as_posix() in vsi_listtree("/vsimem/")

    # Cleanup.
    vsi_unlink(test_file_path)
    assert test_file_path not in vsi_listtree("/vsimem/")


def test_default_layer_name():
    assert _default_layer_name("/path/to/file.gpkg") == "file"
    assert _default_layer_name("file.gpkg") == "file"
    assert _default_layer_name("/path/to/file") == "file"
    # multi-part extensions are stripped completely
    assert _default_layer_name("/path/to/file.gpkg.zip") == "file"
    assert _default_layer_name("/path/to/file.shp.zip") == "file"


def test_validate_atomic_write():
    # valid usage does not raise
    _validate_atomic_write("/path/to/file.gpkg", "GPKG", append=False)

    with pytest.raises(ValueError, match="not supported in combination with append"):
        _validate_atomic_write("/path/to/file.gpkg", "GPKG", append=True)

    with pytest.raises(ValueError, match="only supported for writing to a local"):
        _validate_atomic_write("/vsimem/file.gpkg", "GPKG", append=False)

    with pytest.raises(ValueError, match="only supported for writing to a local"):
        _validate_atomic_write(b"not-a-path", "GPKG", append=False)

    # formats consisting of multiple files cannot be replaced atomically
    with pytest.raises(ValueError, match="not supported for the 'ESRI Shapefile'"):
        _validate_atomic_write("/path/to/file.shp", "ESRI Shapefile", append=False)

    with pytest.raises(ValueError, match="not supported for the 'GeoJSON'"):
        _validate_atomic_write("/path/to/file.geojson", "GeoJSON", append=False)


@pytest.fixture
def mock_list_layers(monkeypatch):
    """Mock the verification of the staging file done by _atomic_publish."""
    import pyogrio._io

    monkeypatch.setattr(
        pyogrio._io, "ogr_list_layers", lambda path: [["layer", "Point"]]
    )


def test_atomic_publish_commit(tmp_path, mock_list_layers):
    target = tmp_path / "test.gpkg"
    target.write_bytes(b"old data")

    with _atomic_publish(str(target), "layer") as staging_path:
        # staging file is created in the same directory with a unique name
        assert Path(staging_path).parent == target.parent
        assert Path(staging_path).name != target.name
        Path(staging_path).write_bytes(b"new data")

    # target is replaced with the new data and no staging files are left behind
    assert target.read_bytes() == b"new data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_new_file(tmp_path, mock_list_layers):
    target = tmp_path / "test.gpkg"

    with _atomic_publish(str(target), "layer") as staging_path:
        Path(staging_path).write_bytes(b"new data")

    assert target.read_bytes() == b"new data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_failure_preserves_target(tmp_path, mock_list_layers):
    target = tmp_path / "test.gpkg"
    target.write_bytes(b"old data")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        with _atomic_publish(str(target), "layer") as staging_path:
            Path(staging_path).write_bytes(b"partial data")
            # SQLite-based formats may leave sidecar files behind
            Path(staging_path + "-wal").write_bytes(b"partial data")
            Path(staging_path + "-shm").write_bytes(b"partial data")
            raise RuntimeError("simulated write failure")

    # target still has its last complete data; staging files are cleaned up
    assert target.read_bytes() == b"old data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_cancel_preserves_target(tmp_path, mock_list_layers):
    target = tmp_path / "test.gpkg"
    target.write_bytes(b"old data")

    with pytest.raises(KeyboardInterrupt):
        with _atomic_publish(str(target), "layer") as staging_path:
            Path(staging_path).write_bytes(b"partial data")
            raise KeyboardInterrupt

    assert target.read_bytes() == b"old data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_verify_failure_preserves_target(tmp_path, monkeypatch):
    import pyogrio._io

    # the written file does not contain the expected layer
    monkeypatch.setattr(
        pyogrio._io, "ogr_list_layers", lambda path: [["other", "Point"]]
    )

    target = tmp_path / "test.gpkg"
    target.write_bytes(b"old data")

    with pytest.raises(DataSourceError, match="atomic write verification failed"):
        with _atomic_publish(str(target), "layer") as staging_path:
            Path(staging_path).write_bytes(b"new data")

    assert target.read_bytes() == b"old data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_replace_failure_preserves_target(
    tmp_path, mock_list_layers, monkeypatch
):
    target = tmp_path / "test.gpkg"
    target.write_bytes(b"old data")

    def failing_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        with _atomic_publish(str(target), "layer") as staging_path:
            Path(staging_path).write_bytes(b"new data")

    assert target.read_bytes() == b"old data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]


def test_atomic_publish_committed_file_not_cleaned(tmp_path, mock_list_layers):
    # a successfully published file is never cleaned up as a staging file,
    # also not when a later publish to the same target fails
    target = tmp_path / "test.gpkg"

    with _atomic_publish(str(target), "layer") as staging_path:
        Path(staging_path).write_bytes(b"committed data")

    assert target.read_bytes() == b"committed data"

    with pytest.raises(RuntimeError, match="simulated write failure"):
        with _atomic_publish(str(target), "layer") as staging_path:
            raise RuntimeError("simulated write failure")

    # the previously committed file is left untouched
    assert target.read_bytes() == b"committed data"
    assert [f.name for f in tmp_path.iterdir()] == ["test.gpkg"]
