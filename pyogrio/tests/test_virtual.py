"""Tests for in-memory multi-file vector datasets."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import shapely

import pyogrio
from pyogrio import (
    VirtualVectorReader,
    VirtualVectorWriter,
    open_virtual_vector_files,
    open_virtual_vector_files_for_write,
    read_dataframe,
    read_info,
    vsi_listtree,
    write_dataframe,
)
from pyogrio.raw import read, write

FIXTURE_DIR = Path(__file__).parent.resolve() / "fixtures" / "naturalearth_lowres"


@pytest.fixture
def shapefile_members():
    return {path.name: path.read_bytes() for path in FIXTURE_DIR.iterdir()}


@pytest.fixture
def points_df():
    import geopandas as gp

    return gp.GeoDataFrame(
        {"name": ["a", "b", "c"], "value": [1, 2, 3]},
        geometry=shapely.points([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]),
        crs="EPSG:4326",
    )


def test_read_virtual_shapefile(shapefile_members):
    with open_virtual_vector_files(shapefile_members) as source:
        assert source.driver == "ESRI Shapefile"
        assert source.layer == "naturalearth_lowres"
        assert source.path.startswith("/vsimem/")
        assert source.path.endswith("naturalearth_lowres.shp")

        df = read_dataframe(source.path, layer=source.layer)
        info = read_info(source.path, layer=source.layer)

    assert len(df) == 177
    assert info["driver"] == "ESRI Shapefile"


def test_virtual_reader_preserves_crs_and_encoding(shapefile_members):
    with open_virtual_vector_files(shapefile_members) as source:
        info = read_info(source.path, layer=source.layer)
        df = read_dataframe(source.path, layer=source.layer)

    # prj member is staged as-is; CRS is preserved
    assert info["crs"] is not None
    assert df.crs is not None

    # cpg member declares ISO-8859-1; GDAL decodes to UTF-8
    assert info["encoding"] == "UTF-8"


def test_virtual_reader_accepts_filelike_members(shapefile_members):
    file_like = {name: BytesIO(data) for name, data in shapefile_members.items()}

    with open_virtual_vector_files(file_like) as source:
        df = read_dataframe(source.path, layer=source.layer)

    assert len(df) == 177

    # the original file-like objects were rewound and remain usable
    for buffer in file_like.values():
        assert buffer.tell() == 0


def test_virtual_reader_accepts_bytearray_members(shapefile_members):
    members = {name: bytearray(data) for name, data in shapefile_members.items()}

    with open_virtual_vector_files(members) as source:
        df = read_dataframe(source.path, layer=source.layer)

    assert len(df) == 177


def test_virtual_reader_cleans_up_on_close(shapefile_members):
    source = open_virtual_vector_files(shapefile_members)
    with source:
        path = source.path
        assert path in vsi_listtree("/vsimem")

    assert vsi_listtree("/vsimem") == []

    with pytest.raises(ValueError, match="closed"):
        source.path

    # close() is idempotent
    source.close()


def test_virtual_reader_cleans_up_on_enter_error(shapefile_members):
    broken = dict(shapefile_members)
    del broken["naturalearth_lowres.shx"]

    with pytest.raises(ValueError, match="missing required member"):
        open_virtual_vector_files(broken).__enter__()

    assert vsi_listtree("/vsimem") == []


def test_virtual_reader_missing_required_member(shapefile_members):
    # removing the main file (.shp) leaves no recognizable multi-file driver
    broken = dict(shapefile_members)
    del broken["naturalearth_lowres.shp"]
    with pytest.raises(ValueError, match="could not determine"):
        open_virtual_vector_files(broken).__enter__()

    # removing a required sibling member while the main file is present fails
    # with an explicit missing-member error
    for removed in ("naturalearth_lowres.shx", "naturalearth_lowres.dbf"):
        broken = dict(shapefile_members)
        del broken[removed]
        with pytest.raises(ValueError, match="missing required member"):
            open_virtual_vector_files(broken).__enter__()


def test_virtual_reader_unknown_driver(shapefile_members):
    members = {"foo.txt": b"hello", "bar.csv": b"a,b\n1,2\n"}
    with pytest.raises(ValueError, match="could not determine"):
        open_virtual_vector_files(members).__enter__()


def test_virtual_reader_rejects_directory_components(shapefile_members):
    members = {f"subdir/{name}": data for name, data in shapefile_members.items()}
    with pytest.raises(ValueError, match="plain file names"):
        open_virtual_vector_files(members).__enter__()


def test_write_files_to_vsimem_rejects_invalid_names():
    from pyogrio._vsi import write_files_to_vsimem

    with pytest.raises(ValueError, match="plain file names"):
        write_files_to_vsimem({"subdir/a.shp": b"1"})

    # invalid input must not leave an in-memory directory behind
    assert vsi_listtree("/vsimem") == []

    with pytest.raises(ValueError, match="at least one member"):
        write_files_to_vsimem({})


def test_virtual_reader_rejects_bad_buffer_type():
    with pytest.raises(TypeError, match="must be bytes"):
        open_virtual_vector_files(
            {"a.shp": "not-bytes", "a.shx": b"", "a.dbf": b""}
        ).__enter__()


def test_virtual_reader_empty_mapping():
    with pytest.raises(ValueError, match="at least one member"):
        open_virtual_vector_files({}).__enter__()


def test_virtual_reader_non_mapping():
    with pytest.raises(TypeError, match="dict"):
        open_virtual_vector_files([("a.shp", b"")]).__enter__()


def test_virtual_reader_layer_selection(shapefile_members):
    with open_virtual_vector_files(shapefile_members) as source:
        # selecting the single existing layer by name and index works
        assert source.layer == "naturalearth_lowres"

    with open_virtual_vector_files(shapefile_members, layer=0) as source:
        assert source.layer == "naturalearth_lowres"

    with open_virtual_vector_files(
        shapefile_members, layer="naturalearth_lowres"
    ) as source:
        df = read_dataframe(source.path, layer=source.layer)
        assert len(df) == 177


def test_virtual_reader_unknown_layer_raises(shapefile_members):
    with pytest.raises(ValueError, match="not found"):
        open_virtual_vector_files(
            shapefile_members, layer="does-not-exist"
        ).__enter__()


def test_virtual_reader_layer_index_out_of_range(shapefile_members):
    with pytest.raises(ValueError, match="out of range"):
        open_virtual_vector_files(shapefile_members, layer=5).__enter__()


def test_virtual_reader_works_with_raw_read(shapefile_members):
    with open_virtual_vector_files(shapefile_members) as source:
        meta, _, geometry, field_data = read(source.path, layer=source.layer)

    assert len(geometry) == 177
    assert meta["geometry_type"] == "Polygon"


def test_virtual_reader_works_with_arrow(shapefile_members):
    pytest.importorskip("pyarrow")
    from pyogrio.raw import open_arrow

    with open_virtual_vector_files(shapefile_members) as source:
        with open_arrow(source.path, layer=source.layer) as (meta, stream):
            import pyarrow as pa

            reader = pa.RecordBatchReader.from_stream(stream)
            table = reader.read_all()

    assert table.num_rows == 177
    # the virtual directory must be gone only after the arrow reader closed
    assert vsi_listtree("/vsimem") == []


def test_virtual_reader_member_names_case_insensitive(shapefile_members):
    members = {name.upper(): data for name, data in shapefile_members.items()}
    with open_virtual_vector_files(members) as source:
        assert source.driver == "ESRI Shapefile"
        df = read_dataframe(source.path, layer=source.layer)
        assert len(df) == 177


def test_write_virtual_shapefile_roundtrip(points_df):
    with open_virtual_vector_files_for_write(
        "ESRI Shapefile", layer="points"
    ) as target:
        assert target.path.startswith("/vsimem/")
        assert target.path.endswith(".shp")
        write_dataframe(
            points_df, target.path, layer=target.layer, driver=target.driver
        )
        members = target.close_and_get_files()

    extensions = {Path(name).suffix.lower() for name in members}
    assert {".shp", ".shx", ".dbf", ".prj"}.issubset(extensions)
    assert vsi_listtree("/vsimem") == []

    # the returned group can be reopened without touching a temp directory
    with open_virtual_vector_files(members) as source:
        result = read_dataframe(source.path, layer=source.layer)
        info = read_info(source.path, layer=source.layer)

    assert len(result) == len(points_df)
    assert result.crs.to_epsg() == 4326
    assert np.array_equal(
        shapely.get_x(result.geometry.values),
        shapely.get_x(points_df.geometry.values),
    )
    assert info["crs"] is not None


def test_write_virtual_shapefile_preserves_encoding(points_df):
    with open_virtual_vector_files_for_write(
        "ESRI Shapefile", layer="points"
    ) as target:
        write_dataframe(
            points_df,
            target.path,
            layer=target.layer,
            driver=target.driver,
            encoding="ISO-8859-1",
        )
        members = target.close_and_get_files()

    cpg_members = [name for name in members if name.lower().endswith(".cpg")]
    assert len(cpg_members) == 1
    assert members[cpg_members[0]].decode().strip() == "ISO-8859-1"


def test_virtual_writer_discards_on_context_exception(points_df):
    from pyogrio.raw import write as raw_write

    # a geometry type that conflicts with the actual data fails inside GDAL,
    # after the in-memory dataset has been created
    line_geometry = shapely.to_wkb(
        shapely.from_wkt(["LINESTRING(0 0, 1 1)", "LINESTRING(2 2, 3 3)",
                          "LINESTRING(4 4, 5 5)"])
    )

    with pytest.raises(Exception):
        with open_virtual_vector_files_for_write(
            "ESRI Shapefile", layer="boom"
        ) as target:
            raw_write(
                target.path,
                line_geometry,
                [points_df["name"].values, points_df["value"].values],
                ["name", "value"],
                layer=target.layer,
                driver=target.driver,
                crs="EPSG:4326",
                geometry_type="Point",
            )

    # no partial package is left behind
    assert vsi_listtree("/vsimem") == []


def test_virtual_writer_does_not_deliver_incomplete_package():
    # open and close without writing anything: required members are missing,
    # so delivering the package must fail and clean up
    target = open_virtual_vector_files_for_write("ESRI Shapefile", layer="empty")
    target.__enter__()
    with pytest.raises(OSError, match="did not produce required member"):
        target.close_and_get_files()

    assert vsi_listtree("/vsimem") == []


def test_virtual_writer_unsupported_driver():
    with pytest.raises(ValueError, match="multi-file"):
        open_virtual_vector_files_for_write("GPKG").__enter__()

    assert vsi_listtree("/vsimem") == []


def test_virtual_writer_close_without_get_files(points_df):
    with open_virtual_vector_files_for_write(
        "ESRI Shapefile", layer="points"
    ) as target:
        write_dataframe(
            points_df, target.path, layer=target.layer, driver=target.driver
        )
        # exiting via context manager discards the data without error
    assert vsi_listtree("/vsimem") == []


def test_virtual_writer_default_layer_name(points_df):
    with open_virtual_vector_files_for_write("ESRI Shapefile") as target:
        assert target.layer
        write_dataframe(
            points_df, target.path, layer=target.layer, driver=target.driver
        )
        members = target.close_and_get_files()

    with open_virtual_vector_files(members) as source:
        assert read_dataframe(source.path, layer=source.layer).shape[0] == 3


def test_write_then_read_raw_roundtrip(points_df):
    with open_virtual_vector_files_for_write(
        "ESRI Shapefile", layer="points"
    ) as target:
        write(
            target.path,
            shapely.to_wkb(points_df.geometry.values),
            [points_df["name"].values, points_df["value"].values],
            ["name", "value"],
            layer=target.layer,
            driver=target.driver,
            crs="EPSG:4326",
            geometry_type="Point",
        )
        members = target.close_and_get_files()

    with open_virtual_vector_files(members) as source:
        meta, _, geometry, field_data = read(source.path, layer=source.layer)

    assert len(geometry) == 3
    assert meta["crs"] is not None


def test_public_classes_repr():
    reader = VirtualVectorReader({})
    assert "unopened" in repr(reader)
    reader._closed = True
    assert "closed" in repr(reader)

    writer = VirtualVectorWriter("ESRI Shapefile")
    assert "ESRI Shapefile" in repr(writer)


def test_public_api_exports():
    assert pyogrio.open_virtual_vector_files is open_virtual_vector_files
    assert (
        pyogrio.open_virtual_vector_files_for_write
        is open_virtual_vector_files_for_write
    )
    assert pyogrio.VirtualVectorReader is VirtualVectorReader
    assert pyogrio.VirtualVectorWriter is VirtualVectorWriter
