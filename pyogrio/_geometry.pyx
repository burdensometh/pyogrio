import warnings
from pyogrio._ogr cimport *
from pyogrio._err cimport *
from pyogrio._err import CPLE_BaseError, NullPointerError
from pyogrio.errors import DataLayerError, GeometryError


# Mapping of OGR integer geometry types to GeoJSON type names.

GEOMETRY_TYPES = {
    wkbUnknown: "Unknown",
    wkbPoint: "Point",
    wkbLineString: "LineString",
    wkbPolygon: "Polygon",
    wkbMultiPoint: "MultiPoint",
    wkbMultiLineString: "MultiLineString",
    wkbMultiPolygon: "MultiPolygon",
    wkbGeometryCollection: "GeometryCollection",
    wkbNone: None,
    wkbLinearRing: "LinearRing",
    # WARNING: Measured types are not supported in GEOS and downstream uses
    # these are stripped automatically to their corresponding 2D / 3D types
    wkbPointM: "PointM",
    wkbLineStringM: "Measured LineString",
    wkbPolygonM: "Measured Polygon",
    wkbMultiPointM: "Measured MultiPoint",
    wkbMultiLineStringM: "Measured MultiLineString",
    wkbMultiPolygonM: "Measured MultiPolygon",
    wkbGeometryCollectionM: "Measured GeometryCollection",
    wkbPointZM: "Measured 3D Point",
    wkbLineStringZM: "Measured 3D LineString",
    wkbPolygonZM: "Measured 3D Polygon",
    wkbMultiPointZM: "Measured 3D MultiPoint",
    wkbMultiLineStringZM: "Measured 3D MultiLineString",
    wkbMultiPolygonZM: "Measured 3D MultiPolygon",
    wkbGeometryCollectionZM: "Measured 3D GeometryCollection",
    wkbPoint25D: "Point Z",
    wkbLineString25D: "LineString Z",
    wkbPolygon25D: "Polygon Z",
    wkbMultiPoint25D: "MultiPoint Z",
    wkbMultiLineString25D: "MultiLineString Z",
    wkbMultiPolygon25D: "MultiPolygon Z",
    wkbGeometryCollection25D: "GeometryCollection Z",
}

GEOMETRY_TYPE_CODES = {v: k for k, v in GEOMETRY_TYPES.items()}

# add additional aliases from 2.5D format
GEOMETRY_TYPE_CODES.update({
    "2.5D Point": wkbPoint25D,
    "2.5D LineString": wkbLineString25D,
    "2.5D Polygon": wkbPolygon25D,
    "2.5D MultiPoint": wkbMultiPoint25D,
    "2.5D MultiLineString": wkbMultiLineString25D,
    "2.5D MultiPolygon": wkbMultiPolygon25D,
    "2.5D GeometryCollection": wkbGeometryCollection25D
})

# 2.5D also represented using negative numbers not enumerated above
GEOMETRY_TYPES.update({
    -2147483647: "Point Z",
    -2147483646: "LineString Z",
    -2147483645: "Polygon Z",
    -2147483644: "MultiPoint Z",
    -2147483643: "MultiLineString Z",
    -2147483642: "MultiPolygon Z",
    -2147483641: "GeometryCollection Z",
})


cdef int get_geometry_field_index(void *ogr_layer, str geometry_name) except -2:
    """Get the index of the geometry field in a layer by its name.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
        The open OGR layer containing the geometry field.
    geometry_name : str
        The name of the geometry field. If None, the first (or only) geometry
        field is used.

    Raises
    ------
    ValueError
        If the layer does not contain a geometry field with ``geometry_name``.

    Returns
    -------
    int
        The index of the geometry field.
    """
    cdef void *ogr_featuredef = NULL
    cdef int geom_field_index
    cdef char *geom_field_c

    with nogil:
        ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)

    if geometry_name is None:
        # Default to the first geometry field. For nonspatial layers, GDAL
        # returns a special wkbNone geometry field definition at index 0.
        return 0

    geom_field_b = geometry_name.encode("UTF-8")
    geom_field_c = geom_field_b
    with nogil:
        geom_field_index = OGR_FD_GetGeomFieldIndex(ogr_featuredef, geom_field_c)

    if geom_field_index < 0:
        raise ValueError(
            f"'geometry_name' must be the name of an existing geometry field; "
            f"got {geometry_name!r}"
        )

    return geom_field_index


cdef str get_geometry_field_name(void *ogr_layer, int geom_field_index):
    """Get the name of a geometry field in a layer by its index.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
        The open OGR layer containing the geometry field.
    geom_field_index : int
        The index of the geometry field.

    Returns
    -------
    str
        The name of the geometry field (an empty string for nonspatial layers or
        drivers that do not support custom geometry field names).
    """
    cdef void *ogr_featuredef = NULL
    cdef void *ogr_geomfielddef = NULL

    with nogil:
        ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)

    # Nonspatial layers do not have any geometry fields
    if OGR_FD_GetGeomFieldCount(ogr_featuredef) == 0:
        return ""

    with nogil:
        ogr_geomfielddef = OGR_FD_GetGeomFieldDefn(
            ogr_featuredef, geom_field_index
        )

    return get_string(OGR_GFld_GetNameRef(ogr_geomfielddef))


cdef str get_geometry_type(void *ogr_layer, int geom_field_index):
    """Get geometry type for a geometry field of a layer.

    Parameters
    ----------
    ogr_layer : pointer to open OGR layer
        The open OGR layer to get the geometry type for.
    geom_field_index : int
        The index of the geometry field on the layer.

    Returns
    -------
    str
        geometry type

    """
    cdef void *ogr_featuredef = NULL
    cdef void *ogr_geomfielddef = NULL
    cdef OGRwkbGeometryType ogr_type
    cdef int geom_field_count

    try:
        with nogil:
            ogr_featuredef = OGR_L_GetLayerDefn(ogr_layer)
        ogr_featuredef = check_pointer(ogr_featuredef)
    except NullPointerError:
        raise DataLayerError("Could not get layer definition")

    except CPLE_BaseError as exc:
        raise DataLayerError(str(exc))

    # Nonspatial layers do not have any geometry fields
    geom_field_count = OGR_FD_GetGeomFieldCount(ogr_featuredef)
    if geom_field_count == 0:
        return None

    with nogil:
        ogr_geomfielddef = OGR_FD_GetGeomFieldDefn(
            ogr_featuredef, geom_field_index
        )
    ogr_geomfielddef = check_pointer(ogr_geomfielddef)

    ogr_type = OGR_GFld_GetType(ogr_geomfielddef)

    if ogr_type not in GEOMETRY_TYPES:
        raise GeometryError(f"Geometry type is not supported: {ogr_type}")

    if OGR_GT_HasM(ogr_type):
        original_type = GEOMETRY_TYPES[ogr_type]

        # Downgrade the type to 2D / 3D
        ogr_type = OGR_GT_SetModifier(ogr_type, OGR_GT_HasZ(ogr_type), 0)

        # TODO: review; this might be annoying...
        warnings.warn(
            "Measured (M) geometry types are not supported. "
            f"Original type '{original_type}' "
            f"is converted to '{GEOMETRY_TYPES[ogr_type]}'")

    return GEOMETRY_TYPES[ogr_type]


cdef OGRwkbGeometryType get_geometry_type_code(str geometry_type) except *:
    """Get geometry type code for string geometry type.

    Parameters
    ----------
    geometry_type : str
        Geometry type as a string.

    Returns
    -------
    int
        geometry type code

    """
    if geometry_type not in GEOMETRY_TYPE_CODES:
        raise GeometryError(f"Geometry type is not supported: {geometry_type}")

    return GEOMETRY_TYPE_CODES[geometry_type]
