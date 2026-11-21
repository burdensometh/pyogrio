"""Tests for the dual-column (UTC + original offset) datetime contract."""

import numpy as np

from pyogrio import read_dataframe, write_dataframe
from pyogrio._datetime import DATETIME_LAYOUT_UTC_OFFSET
from pyogrio.raw import open_arrow, read, read_arrow, write
from pyogrio.tests.conftest import requires_pyarrow_api

import pytest

try:
    import pandas as pd
    import pyarrow as pa

    from geopandas.testing import assert_geodataframe_equal
except ImportError:
    pass


LAYOUT = DATETIME_LAYOUT_UTC_OFFSET
EXT = "_offset"


def _assert_classic_arrays(meta, field_data):
    names = list(meta["fields"])
    data = dict(zip(names, field_data, strict=True))
    assert "observed" in names
    assert f"observed{EXT}" in names

    observed = data["observed"]
    offsets = data[f"observed{EXT}"]
    assert observed.dtype == "datetime64[ms]"
    assert offsets.dtype == "timedelta64[ms]"

    # DST fall-back: the two 01:30 readings are distinct instants
    assert observed[0] == np.datetime64("2024-11-03T05:30:00.000")
    assert observed[1] == np.datetime64("2024-11-03T06:30:00.000")
    assert offsets[0] == np.timedelta64(-4, "h")
    assert offsets[1] == np.timedelta64(-5, "h")

    # explicit zero offset (UTC) is kept distinct from an unknown offset
    assert observed[2] == np.datetime64("2024-01-01T00:00:00.000")
    assert offsets[2] == np.timedelta64(0, "ms")

    # a datetime without an offset is an unknown time zone, read as UTC
    assert observed[3] == np.datetime64("2024-01-01T00:00:00.000")
    assert np.isnat(offsets[3])

    # null stays null in both columns
    assert np.isnat(observed[4])
    assert np.isnat(offsets[4])

    return observed, offsets


def test_datetime_layout_classic(datetime_offset_file):
    default_meta, _, _, _ = read(datetime_offset_file)
    meta, _, _, field_data = read(datetime_offset_file, datetime_layout=LAYOUT)

    assert list(meta["fields"]) == ["observed", f"observed{EXT}"]
    assert list(meta["ogr_types"]) == ["OFTDateTime", "OFTDateTimeOffset"]
    assert meta["datetime_layout"] == LAYOUT
    # the default behaviour is unchanged
    assert list(default_meta["fields"]) == ["observed"]

    _assert_classic_arrays(meta, field_data)


def test_datetime_layout_classic_batches(datetime_offset_file):
    """Offsets do not shift when the rows are read in several chunks."""
    full = read(datetime_offset_file, datetime_layout=LAYOUT)[3]
    for skip, max_features in [(1, None), (2, None), (0, 2), (1, 2), (3, 1)]:
        chunked = read(
            datetime_offset_file,
            datetime_layout=LAYOUT,
            skip_features=skip,
            max_features=max_features,
        )[3]
        for full_field, chunked_field in zip(full, chunked, strict=True):
            start = skip
            stop = None if max_features is None else skip + max_features
            assert np.array_equal(full_field[start:stop], chunked_field, equal_nan=True)


def test_datetime_layout_classic_fids(datetime_offset_file):
    meta, _, _, field_data = read(
        datetime_offset_file, datetime_layout=LAYOUT, fids=[1, 4]
    )
    data = dict(zip(meta["fields"], field_data, strict=True))
    assert data["observed"][0] == np.datetime64("2024-11-03T06:30:00.000")
    assert data[f"observed{EXT}"][0] == np.timedelta64(-5, "h")
    assert np.isnat(data["observed"][1])
    assert np.isnat(data[f"observed{EXT}"][1])


def test_datetime_layout_classic_columns(datetime_offset_file):
    meta, _, _, field_data = read(
        datetime_offset_file, datetime_layout=LAYOUT, columns=["observed"]
    )
    assert list(meta["fields"]) == ["observed", f"observed{EXT}"]
    _assert_classic_arrays(meta, field_data)


def test_datetime_layout_with_datetime_as_string_raises(datetime_offset_file):
    with pytest.raises(ValueError, match="cannot be used together"):
        read(datetime_offset_file, datetime_layout=LAYOUT, datetime_as_string=True)


def test_datetime_layout_invalid(datetime_offset_file):
    with pytest.raises(ValueError, match="'datetime_layout' must be one of"):
        read(datetime_offset_file, datetime_layout="bogus")


def test_datetime_layout_roundtrip_classic(tmp_path, datetime_offset_file):
    meta, _, geometry, field_data = read(datetime_offset_file, datetime_layout=LAYOUT)
    output = tmp_path / "test.gpkg"
    write(
        output,
        geometry,
        field_data,
        meta["fields"],
        geometry_type=meta["geometry_type"],
        crs=meta["crs"],
        datetime_layout=LAYOUT,
    )

    result_meta, _, _, result = read(output, datetime_layout=LAYOUT)
    assert list(result_meta["fields"]) == ["observed", f"observed{EXT}"]
    _assert_classic_arrays(result_meta, result)

    # what is stored on disk keeps the local time and its offset
    stored = read(output, datetime_as_string=True)[3][0]
    assert list(stored[:3]) == [
        "2024/11/03 01:30:00-04",
        "2024/11/03 01:30:00-05",
        "2024/01/01 00:00:00+00",
    ]


def test_datetime_layout_write_rejects_precision(tmp_path, datetime_offset_file):
    meta, _, geometry, field_data = read(datetime_offset_file, datetime_layout=LAYOUT)
    # 20 minutes is not a multiple of 15, so GDAL cannot store it
    bad_offsets = field_data[1].copy()
    bad_offsets[0] = np.timedelta64(20, "m")
    with pytest.raises(ValueError, match="not a multiple of 15 minutes"):
        write(
            tmp_path / "test.gpkg",
            geometry,
            [field_data[0], bad_offsets],
            meta["fields"],
            geometry_type=meta["geometry_type"],
            datetime_layout=LAYOUT,
        )


def test_datetime_layout_write_rejects_sub_millisecond(tmp_path, datetime_offset_file):
    meta, _, geometry, field_data = read(datetime_offset_file, datetime_layout=LAYOUT)
    precise = field_data[0].astype("datetime64[ns]")
    precise[0] = np.datetime64("2024-11-03T05:30:00.000000500")
    with pytest.raises(ValueError, match="more precise than milliseconds"):
        write(
            tmp_path / "test.gpkg",
            geometry,
            [precise, field_data[1]],
            meta["fields"],
            geometry_type=meta["geometry_type"],
            datetime_layout=LAYOUT,
        )


def test_datetime_layout_write_rejects_driver_without_datetime(tmp_path):
    from pyogrio.errors import DataSourceError

    fields = ["observed", f"observed{EXT}"]
    field_data = [
        np.array(["2024-01-01T00:00:00.000"], dtype="datetime64[ms]"),
        np.array([np.timedelta64(0, "m")], dtype="timedelta64[ms]"),
    ]
    with pytest.raises(DataSourceError, match="cannot store date-times"):
        write(
            tmp_path / "test.shp",
            None,
            field_data,
            fields,
            geometry_type="Point",
            datetime_layout=LAYOUT,
        )


@requires_pyarrow_api
def test_datetime_layout_arrow(datetime_offset_file):
    meta, table = read_arrow(datetime_offset_file, datetime_layout=LAYOUT)

    assert list(meta["fields"]) == ["observed", f"observed{EXT}"]
    assert table.schema.field("observed").type == pa.timestamp("ms", tz="UTC")
    assert table.schema.field(f"observed{EXT}").type == pa.duration("ms")

    observed = table.column("observed").to_pylist()
    offsets = table.column(f"observed{EXT}").to_pylist()
    assert observed[0].isoformat() == "2024-11-03T05:30:00+00:00"
    assert observed[1].isoformat() == "2024-11-03T06:30:00+00:00"
    assert offsets[0].total_seconds() == -4 * 3600
    assert offsets[1].total_seconds() == -5 * 3600
    assert offsets[2].total_seconds() == 0
    assert offsets[3] is None
    assert observed[4] is None

    # the classic and Arrow results describe the same instants
    classic = read(datetime_offset_file, datetime_layout=LAYOUT)[3][0]
    assert list(classic[:4]) == [
        np.datetime64("2024-11-03T05:30:00.000"),
        np.datetime64("2024-11-03T06:30:00.000"),
        np.datetime64("2024-01-01T00:00:00.000"),
        np.datetime64("2024-01-01T00:00:00.000"),
    ]
    assert [value.replace(tzinfo=None) for value in observed[:4]] == [
        value.astype("datetime64[ms]").astype("datetime64[us]").item()
        for value in classic[:4]
    ]


@requires_pyarrow_api
def test_datetime_layout_arrow_matches_classic_for_fids(datetime_offset_file):
    _, table = read_arrow(datetime_offset_file, datetime_layout=LAYOUT, fids=[1, 4])
    assert table.column("observed").to_pylist()[0].isoformat() == (
        "2024-11-03T06:30:00+00:00"
    )
    assert table.column(f"observed{EXT}").to_pylist()[0].total_seconds() == -5 * 3600
    assert table.column("observed").to_pylist()[1] is None


@requires_pyarrow_api
@pytest.mark.skipif(
    __import__("pyogrio").__gdal_version__ < (3, 11, 0),
    reason="open_arrow needs GDAL >= 3.11 for string datetimes",
)
def test_datetime_layout_open_arrow(datetime_offset_file):
    with open_arrow(datetime_offset_file, datetime_layout=LAYOUT, use_pyarrow=True) as (
        meta,
        reader,
    ):
        assert list(meta["fields"]) == ["observed", f"observed{EXT}"]
        table = pa.Table.from_batches(list(reader))

    assert table.schema.field("observed").type == pa.timestamp("ms", tz="UTC")
    assert table.column(f"observed{EXT}").to_pylist()[1].total_seconds() == -5 * 3600


@pytest.mark.parametrize("use_arrow", [False, True])
def test_read_dataframe_datetime_layout(datetime_offset_file, use_arrow):
    pytest.importorskip("geopandas")
    df = read_dataframe(
        datetime_offset_file, datetime_layout=LAYOUT, use_arrow=use_arrow
    )

    assert list(df.columns) == ["observed", f"observed{EXT}", "geometry"]
    assert isinstance(df["observed"].dtype, pd.DatetimeTZDtype)
    assert str(df["observed"].dtype.tz) == "UTC"
    assert df["observed_offset"].dtype == "timedelta64[ms]"

    assert df["observed"].dt.tz_convert("UTC").tolist()[:2] == list(
        pd.to_datetime(
            ["2024-11-03T05:30:00Z", "2024-11-03T06:30:00Z"], utc=True
        ).as_unit("ms")
    )
    assert df["observed_offset"].iloc[0] == pd.Timedelta(hours=-4)
    assert df["observed_offset"].iloc[1] == pd.Timedelta(hours=-5)
    assert df["observed_offset"].iloc[2] == pd.Timedelta(0)
    assert pd.isna(df["observed_offset"].iloc[3])
    assert pd.isna(df["observed"].iloc[4])


@pytest.mark.parametrize("use_arrow_read", [False, True])
@pytest.mark.parametrize("use_arrow_write", [False, True])
def test_write_dataframe_datetime_layout_roundtrip(
    tmp_path, datetime_offset_file, use_arrow_read, use_arrow_write
):
    pytest.importorskip("geopandas")
    if use_arrow_write:
        pytest.importorskip("pyarrow")
        from pyogrio._compat import HAS_ARROW_WRITE_API

        if not HAS_ARROW_WRITE_API:
            pytest.skip("GDAL >= 3.8 required to write using Arrow")

    df = read_dataframe(
        datetime_offset_file, datetime_layout=LAYOUT, use_arrow=use_arrow_read
    )
    output = tmp_path / "test.gpkg"
    write_dataframe(df, output, datetime_layout=LAYOUT, use_arrow=use_arrow_write)

    assert_geodataframe_equal(
        read_dataframe(output, datetime_layout=LAYOUT), df, check_dtype=False
    )


def test_read_dataframe_datetime_layout_matches_default_instants(
    datetime_offset_file,
):
    """The UTC instants equal what the default (mixed_offsets_as_utc) returns."""
    pytest.importorskip("geopandas")
    df_default = read_dataframe(datetime_offset_file)
    df_dual = read_dataframe(datetime_offset_file, datetime_layout=LAYOUT)

    # compare the non-null instants
    default_utc = df_default["observed"].dt.tz_convert("UTC")
    dual_utc = df_dual["observed"]
    mask = default_utc.notna()
    assert list(default_utc[mask]) == list(dual_utc[mask])


def test_datetime_layout_write_keeps_unknown_offset(tmp_path, datetime_offset_file):
    """An unknown offset stays unknown after a roundtrip, not explicit UTC."""
    pytest.importorskip("geopandas")
    df = read_dataframe(datetime_offset_file, datetime_layout=LAYOUT)
    write_dataframe(df, tmp_path / "test.gpkg", datetime_layout=LAYOUT)
    result = read_dataframe(tmp_path / "test.gpkg", datetime_layout=LAYOUT)

    assert pd.isna(result["observed_offset"].iloc[3])
    assert result["observed_offset"].iloc[2] == pd.Timedelta(0)


@requires_pyarrow_api
def test_write_arrow_combines_utc_and_offset(datetime_offset_file):
    """A UTC timestamp + duration pair becomes a datetime string with offsets."""
    from pyogrio.raw import _arrow_combine_datetime_layout

    utc = pa.array(
        np.array(
            ["2024-11-03T05:30:00.000", "2024-01-01T00:00:00.000"],
            dtype="datetime64[ms]",
        )
    ).cast(pa.timestamp("ms", tz="UTC"))
    offsets = pa.array(
        np.array(
            [np.timedelta64(-4, "h"), np.timedelta64("NaT")], dtype="timedelta64[ms]"
        )
    )
    table = pa.table({"observed": utc, "observed_offset": offsets, "x": [1, 2]})

    result = _arrow_combine_datetime_layout(table, "GPKG")

    assert result.column_names == ["observed", "x"]
    assert pa.types.is_string(result.schema.field("observed").type)
    assert result.schema.field("observed").metadata[b"GDAL:OGR:type"] == b"DateTime"
    assert result.column("observed").to_pylist() == [
        "2024-11-03T01:30:00.000-04:00",
        "2024-01-01T00:00:00.000",
    ]


@requires_pyarrow_api
def test_write_arrow_ignores_non_duration_offset_column():
    """A real column that merely ends with ``_offset`` is left alone."""
    from pyogrio.raw import _arrow_combine_datetime_layout

    utc = pa.array(np.array(["2024-01-01T00:00:00.000"], dtype="datetime64[ms]")).cast(
        pa.timestamp("ms", tz="UTC")
    )
    table = pa.table({"observed": utc, "observed_offset": pa.array(["note"])})

    result = _arrow_combine_datetime_layout(table, "GPKG")
    assert result.column_names == ["observed", "observed_offset"]
    assert pa.types.is_timestamp(result.schema.field("observed").type)


def test_write_arrow_rejects_driver_without_datetime():
    """write_arrow rejects drivers without a DateTime field type."""
    from pyogrio.errors import DataSourceError
    from pyogrio.raw import _arrow_combine_datetime_layout

    pytest.importorskip("pyarrow")
    utc = pa.array(np.array(["2024-01-01T00:00:00.000"], dtype="datetime64[ms]")).cast(
        pa.timestamp("ms", tz="UTC")
    )
    table = pa.table(
        {
            "observed": utc,
            "observed_offset": pa.array(
                np.array([np.timedelta64(0, "m")], dtype="timedelta64[ms]")
            ),
        }
    )

    with pytest.raises(DataSourceError, match="cannot store date-times"):
        _arrow_combine_datetime_layout(table, "ESRI Shapefile")
