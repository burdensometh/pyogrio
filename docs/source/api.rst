.. py:currentmodule:: pyogrio

API reference
=============

Core
----

.. automodule:: pyogrio
   :members: detect_write_driver, get_gdal_config_option, list_drivers, list_drivers_details, list_layers, read_bounds, read_info, set_gdal_config_options, vsi_curl_clear_cache, vsi_listtree, vsi_rmtree, vsi_unlink


..
   For the special attributes/dunder attributes, the inline docstrings weren't
   picked up by autodoc, so they are documented explicitly here.

.. py:attribute:: __version__

   The pyogrio version (`str`).

.. py:attribute:: __gdal_version__

   The GDAL version used by pyogrio (`tuple` of `int`).

.. py:attribute:: __gdal_version_string__

   The GDAL version used by pyogrio (`str`).

.. py:attribute:: __gdal_geos_version__

   The version of GEOS used by GDAL (`tuple` of `int`).


GeoPandas integration
---------------------

.. autofunction:: pyogrio.read_dataframe
.. autofunction:: pyogrio.write_dataframe

Arrow integration
-----------------

.. autofunction:: pyogrio.read_arrow
.. autofunction:: pyogrio.open_arrow
.. autofunction:: pyogrio.write_arrow

In-memory multi-file datasets
-----------------------------

Some vector formats (e.g. ESRI Shapefile) consist of several sibling files
with distinct semantics (``.shp``, ``.shx``, ``.dbf``, ``.prj``, ``.cpg``).
These helpers stage an explicit mapping of member name to bytes buffer in a
GDAL ``/vsimem/`` directory, without writing the members to a temporary
directory. The virtual data is removed when the reader or writer is closed.

.. autofunction:: pyogrio.open_virtual_vector_files
.. autoclass:: pyogrio.VirtualVectorReader
   :members: path, driver, layer, close
.. autofunction:: pyogrio.open_virtual_vector_files_for_write
.. autoclass:: pyogrio.VirtualVectorWriter
   :members: path, driver, layer, close, close_and_get_files
