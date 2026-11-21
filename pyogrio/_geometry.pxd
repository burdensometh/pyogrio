from pyogrio._ogr cimport *

cdef int get_geometry_field_index(void *ogr_layer, str geometry_name) except -2
cdef str get_geometry_field_name(void *ogr_layer, int geom_field_index)
cdef str get_geometry_type(void *ogr_layer, int geom_field_index)
cdef OGRwkbGeometryType get_geometry_type_code(str geometry_type) except *
