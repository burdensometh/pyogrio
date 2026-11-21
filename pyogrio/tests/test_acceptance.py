"""Tests for the geometry acceptance pipeline (pyogrio.acceptance)."""

import sqlite3

import numpy as np

import pyogrio
from pyogrio.acceptance import (
    ALL_STATUSES,
    STATUS_CORRUPT,
    STATUS_EMPTY,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_REPAIRED,
    STATUS_VALID,
    accept_geometries,
    accept_read_result,
    read_accepted,
)
from pyogrio.raw import read, write

import pytest

shapely = pytest.importorskip("shapely", minversion="2.0.0")


VALID_WKT = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
# self-intersecting bowtie: invalid, repaired to a MultiPolygon
BOWTIE_WKT = "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))"
# degenerate ring: invalid, collapses to empty when repaired
SPIKE_WKT = "POLYGON ((0 0, 1 0, 1 0, 0 0))"
EMPTY_WKT = "POLYGON EMPTY"
CORRUPT_WKB = b"\x01\x03\x00\x00\x00garbagegarbage"


def _wkb(*wkts):
    return np.array(
        [
            None if wkt is None else shapely.to_wkb(shapely.from_wkt(wkt))
            for wkt in wkts
        ],
        dtype=object,
    )


@pytest.fixture
def mixed_wkb():
    """WKB array covering all six acceptance statuses."""
    return np.array(
        [
            shapely.to_wkb(shapely.from_wkt(VALID_WKT)),
            shapely.to_wkb(shapely.from_wkt(BOWTIE_WKT)),
            shapely.to_wkb(shapely.from_wkt(SPIKE_WKT)),
            shapely.to_wkb(shapely.from_wkt(EMPTY_WKT)),
            None,
            CORRUPT_WKB,
        ],
        dtype=object,
    )


@pytest.fixture
def supplier_file(tmp_path):
    """GeoPackage as delivered by the supplier: valid, repairable and
    unrepairable polygons, an empty geometry and a missing geometry."""
    path = tmp_path / "supplier.gpkg"
    geometry = _wkb(VALID_WKT, BOWTIE_WKT, SPIKE_WKT, EMPTY_WKT, None)
    values = np.array([10, 20, 30, 40, 50])
    names = np.array(["valid", "bowtie", "spike", "empty", "none"])
    write(
        path,
        geometry,
        [values, names],
        fields=["value", "name"],
        geometry_type="Polygon",
        crs="EPSG:4326",
        layer="supplier",
        spatial_index=False,
    )
    return path


# ---------------------------------------------------------------------------
# accept_geometries
# ---------------------------------------------------------------------------


def test_accept_geometries_classification(mixed_wkb):
    result = accept_geometries(mixed_wkb, fids=[1, 2, 3, 4, 5, 6])

    assert result.report.status.tolist() == [
        STATUS_VALID,
        STATUS_REPAIRED,
        STATUS_INVALID,
        STATUS_EMPTY,
        STATUS_MISSING,
        STATUS_CORRUPT,
    ]
    # only valid and repaired geometries are accepted by default
    assert result.keep_mask.tolist() == [True, True, False, False, False, False]
    assert len(result.geometry) == 2

    # the valid geometry passes through unmodified
    assert result.geometry[0] == mixed_wkb[0]

    # the repaired geometry is valid and changed type Polygon -> MultiPolygon
    repaired = shapely.from_wkb(result.geometry[1])
    assert repaired.geom_type == "MultiPolygon"
    assert shapely.is_valid(repaired)
    assert not shapely.is_empty(repaired)

    # parsed geometries are available for all parseable inputs
    assert shapely.is_missing(result.parsed_geometries[[4, 5]]).all()
    assert result.parsed_geometries[1].geom_type == "MultiPolygon"


def test_accept_geometries_report(mixed_wkb):
    fids = np.array([101, 102, 103, 104, 105, 106])
    result = accept_geometries(mixed_wkb, fids=fids)
    report = result.report

    assert len(report) == 6
    assert report.n_total == 6
    assert report.n_kept == 2
    assert report.n_rejected == 4
    assert report.n_repaired == 1
    assert report.counts() == {
        "valid": 1,
        "repaired": 1,
        "empty": 1,
        "missing": 1,
        "corrupt": 1,
        "invalid": 1,
    }
    assert set(report.counts()) == set(ALL_STATUSES)

    # original FIDs are preserved for adjusted and rejected features
    assert report.repaired_fids.tolist() == [102]
    assert report.adjusted_fids.tolist() == [102]
    assert report.rejected_fids.tolist() == [103, 104, 105, 106]
    assert report.kept_fids.tolist() == [101, 102]
    assert report.fids_with_status(STATUS_EMPTY, STATUS_MISSING).tolist() == [104, 105]

    # reasons are recorded for non-valid features
    assert report.reasons[0] is None
    assert "Self-intersection" in report.reasons[1]
    assert "could not be repaired" in report.reasons[2]
    assert "WKB parse error" in report.reasons[5]

    # report rows align with the input rows (and hence attribute columns)
    d = report.to_dict()
    assert set(d) == {"row_index", "fid", "status", "kept", "reason"}
    assert d["row_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert d["fid"].tolist() == fids.tolist()
    assert d["kept"].tolist() == result.keep_mask.tolist()


def test_accept_geometries_report_without_fids(mixed_wkb):
    report = accept_geometries(mixed_wkb).report
    assert report.fids is None
    # falls back to positional row indices
    assert report.repaired_fids.tolist() == [1]
    assert report.rejected_fids.tolist() == [2, 3, 4, 5]
    assert report.to_dict()["fid"].tolist() == [None] * 6


def test_accept_geometries_fids_length_mismatch(mixed_wkb):
    with pytest.raises(ValueError, match="same length"):
        accept_geometries(mixed_wkb, fids=[1, 2])


def test_accept_geometries_invalid_repair_method(mixed_wkb):
    with pytest.raises(ValueError, match="repair_method"):
        accept_geometries(mixed_wkb, repair_method="buffer0")


def test_accept_geometries_no_repair(mixed_wkb):
    result = accept_geometries(mixed_wkb, repair=False)
    # repairable geometry is rejected like the unrepairable one
    assert result.report.status.tolist()[1:3] == [STATUS_INVALID, STATUS_INVALID]
    assert result.keep_mask.tolist() == [True, False, False, False, False, False]


def test_accept_geometries_linework(mixed_wkb):
    result = accept_geometries(mixed_wkb, repair_method="linework")
    assert result.report.status[1] == STATUS_REPAIRED
    assert shapely.is_valid(shapely.from_wkb(result.geometry[1]))


def test_accept_geometries_declared_keep(mixed_wkb):
    result = accept_geometries(
        mixed_wkb,
        keep_empty=True,
        keep_missing=True,
        keep_invalid=True,
        keep_corrupt=True,
    )
    # everything is kept, but the status still reflects what each geometry is
    assert result.keep_mask.all()
    assert len(result.geometry) == 6
    # non-repaired geometries pass through unmodified
    assert result.geometry[3] == mixed_wkb[3]
    assert result.geometry[4] is None
    assert result.geometry[5] == CORRUPT_WKB


def test_accept_geometries_empty_input():
    result = accept_geometries(np.array([], dtype=object))
    assert len(result.geometry) == 0
    assert len(result.report) == 0
    assert result.report.counts() == dict.fromkeys(ALL_STATUSES, 0)


def test_accept_geometries_z_preserved():
    bowtie_z = "POLYGON Z ((0 0 1, 2 2 2, 2 0 3, 0 2 4, 0 0 1))"
    result = accept_geometries(_wkb(bowtie_z))
    assert result.report.status[0] == STATUS_REPAIRED
    repaired = shapely.from_wkb(result.geometry[0])
    assert repaired.geom_type == "MultiPolygon"
    assert shapely.has_z(repaired)


# ---------------------------------------------------------------------------
# accept_read_result
# ---------------------------------------------------------------------------


def test_accept_read_result_alignment(supplier_file):
    result = accept_read_result(read(supplier_file, return_fids=True))

    # only the valid and repaired features enter the accepted result
    assert result.fids.tolist() == [1, 2]
    assert len(result.geometry) == 2
    # attribute columns are filtered with the same mask
    assert result.field_data[0].tolist() == [10, 20]
    assert result.field_data[1].tolist() == ["valid", "bowtie"]

    # the report covers all input features, in input order
    assert result.report.fids.tolist() == [1, 2, 3, 4, 5]
    assert result.report.status.tolist() == [
        STATUS_VALID,
        STATUS_REPAIRED,
        STATUS_INVALID,
        STATUS_EMPTY,
        STATUS_MISSING,
    ]
    assert result.report.repaired_fids.tolist() == [2]
    assert result.report.rejected_fids.tolist() == [3, 4, 5]


def test_accept_read_result_quarantine(supplier_file):
    result = accept_read_result(read(supplier_file, return_fids=True), quarantine=True)

    rejected = result.rejected
    assert rejected is not None
    assert len(rejected) == 3
    # original FIDs, unmodified WKB and attribute values are preserved
    assert rejected.fids.tolist() == [3, 4, 5]
    assert rejected.field_data[0].tolist() == [30, 40, 50]
    assert rejected.field_data[1].tolist() == ["spike", "empty", "none"]
    assert rejected.status.tolist() == [STATUS_INVALID, STATUS_EMPTY, STATUS_MISSING]
    assert rejected.geometry[1] is not None  # empty polygon WKB preserved
    assert rejected.geometry[2] is None  # missing geometry preserved as None
    assert rejected.reasons[0] is not None


def test_accept_read_result_no_quarantine(supplier_file):
    result = accept_read_result(read(supplier_file, return_fids=True))
    assert result.rejected is None
    # rejected FIDs are still available from the report
    assert result.report.rejected_fids.tolist() == [3, 4, 5]


def test_accept_read_result_updates_geometry_type(supplier_file):
    meta, *_ = read(supplier_file, return_fids=True)
    assert meta["geometry_type"] == "Polygon"

    result = accept_read_result(read(supplier_file, return_fids=True))
    # repaired bowtie became a MultiPolygon: mixed Polygon/MultiPolygon
    # is declared as MultiPolygon so it can be written out
    assert result.meta["geometry_type"] == "MultiPolygon"
    # the input meta dict is not mutated
    assert meta["geometry_type"] == "Polygon"


def test_accept_read_result_derives_z_type():
    meta = {
        "crs": "EPSG:4326",
        "fields": np.array(["value"]),
        "dtypes": np.array(["int64"]),
        "ogr_types": np.array(["OFTInteger64"]),
        "ogr_subtypes": np.array(["OFSTNone"]),
        "encoding": "UTF-8",
        "geometry_type": "Polygon Z",
    }
    wkbs = _wkb(
        "POLYGON Z ((0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1))",
        "POLYGON Z ((0 0 1, 2 2 2, 2 0 3, 0 2 4, 0 0 1))",
    )
    result = accept_read_result((meta, np.array([1, 2]), wkbs, (np.array([1, 2]),)))
    assert result.meta["geometry_type"] == "MultiPolygon Z"
    assert shapely.has_z(shapely.from_wkb(result.geometry)).all()


def test_accept_read_result_without_fids(supplier_file):
    with pytest.warns(UserWarning, match="does not include FIDs"):
        result = accept_read_result(read(supplier_file))
    assert result.fids is None
    assert result.report.fids is None
    assert result.report.repaired_fids.tolist() == [1]


def test_accept_read_result_no_geometry(supplier_file):
    with pytest.raises(ValueError, match="does not contain geometries"):
        accept_read_result(read(supplier_file, read_geometry=False))


def test_accept_read_result_unpacking(supplier_file):
    result = read_accepted(supplier_file)
    meta, fids, geometry, field_data = result
    assert meta is result.meta
    assert fids is result.fids
    assert geometry is result.geometry
    assert field_data is result.field_data
    assert result.as_tuple() == (meta, fids, geometry, field_data)


# ---------------------------------------------------------------------------
# read_accepted
# ---------------------------------------------------------------------------


def test_read_accepted(supplier_file):
    result = read_accepted(supplier_file, quarantine=True)

    assert isinstance(result.meta, dict)
    assert result.meta["crs"] == "EPSG:4326"
    assert result.meta["geometry_type"] == "MultiPolygon"
    assert result.fids.tolist() == [1, 2]
    assert result.report.counts() == {
        "valid": 1,
        "repaired": 1,
        "empty": 1,
        "missing": 1,
        "corrupt": 0,
        "invalid": 1,
    }
    assert result.rejected.fids.tolist() == [3, 4, 5]


def test_read_accepted_passes_read_kwargs(supplier_file):
    result = read_accepted(supplier_file, max_features=2)
    assert result.report.fids.tolist() == [1, 2]
    assert result.fids.tolist() == [1, 2]

    result = read_accepted(supplier_file, where="value >= 40")
    assert result.report.status.tolist() == [STATUS_EMPTY, STATUS_MISSING]


def test_read_accepted_forces_fids(supplier_file):
    with pytest.warns(UserWarning, match="always reads FIDs"):
        result = read_accepted(supplier_file, return_fids=False)
    assert result.fids is not None
    assert result.report.fids is not None


def test_read_accepted_keep_missing(supplier_file):
    result = read_accepted(supplier_file, keep_missing=True)
    # the feature without geometry is kept as declared
    assert result.fids.tolist() == [1, 2, 5]
    assert result.geometry[2] is None
    assert result.field_data[1].tolist() == ["valid", "bowtie", "none"]


def test_read_accepted_all_rejected(tmp_path):
    path = tmp_path / "all_invalid.gpkg"
    write(
        path,
        _wkb(SPIKE_WKT),
        [np.array([1])],
        fields=["value"],
        geometry_type="Polygon",
        crs="EPSG:4326",
        layer="invalid",
        spatial_index=False,
    )

    result = read_accepted(path)
    assert len(result.geometry) == 0
    assert len(result.field_data[0]) == 0
    assert result.report.n_rejected == 1
    # no geometries left to derive from: original layer type is kept
    assert result.meta["geometry_type"] == "Polygon"


def test_read_accepted_top_level():
    assert pyogrio.read_accepted is read_accepted


@pytest.mark.parametrize("ext", [".gpkg", ".shp", ".geojson"])
def test_accepted_result_write_roundtrip(tmp_path, supplier_file, ext):
    """The accepted result (with repaired, type-changed geometries) can be
    written out and read back with attributes still aligned."""
    result = read_accepted(supplier_file, quarantine=True)

    out = tmp_path / f"cleaned{ext}"
    kwargs = {"spatial_index": False} if ext == ".gpkg" else {}
    write(out, result.geometry, result.field_data, **kwargs, **result.meta)

    meta, _, geometry, field_data = read(out)
    # GPKG preserves the declared MultiPolygon layer type; shapefiles have a
    # single polygon shape type and GeoJSON reports mixed types as Unknown
    assert (
        meta["geometry_type"]
        == {
            ".gpkg": "MultiPolygon",
            ".shp": "Polygon",
            ".geojson": "Unknown",
        }[ext]
    )
    assert len(geometry) == 2
    geoms = shapely.from_wkb(geometry)
    assert shapely.is_valid(geoms).all()
    assert not shapely.is_empty(geoms).any()
    # repaired geometry stays a MultiPolygon; for drivers without mixed
    # single/multi support the valid Polygon was promoted as well
    assert geoms[-1].geom_type == "MultiPolygon"
    # attributes remain aligned with the geometries
    assert field_data[0].tolist() == [10, 20]
    assert field_data[1].tolist() == ["valid", "bowtie"]


def test_corrupt_blob_surfaces_as_missing(tmp_path, supplier_file):
    """A geometry blob corrupted in the supplier file is nulled out by GDAL
    and classified as missing by the pipeline."""
    # corrupt the blob of the first feature (valid GPKG header, garbage WKB)
    con = sqlite3.connect(supplier_file)
    (blob,) = con.execute("SELECT hex(geom) FROM supplier WHERE fid = 1").fetchone()
    con.execute(
        "UPDATE supplier SET geom = ? WHERE fid = 1",
        (bytes.fromhex(blob[:16]) + CORRUPT_WKB,),
    )
    con.commit()
    con.close()

    result = read_accepted(supplier_file)
    assert result.report.status[0] == STATUS_MISSING
    assert 1 in result.report.rejected_fids
