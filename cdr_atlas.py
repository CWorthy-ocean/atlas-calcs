"""
Utilities for analyzing atlas datasets within model grid boundaries.

This module provides functions and classes for:
- Retrieving and caching atlas data from S3
- Identifying points and polygons within model grid boundaries
- Integrating FG_CO2 data over spatial and temporal dimensions
"""

# Standard library imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs

import xarray as xr
import pop_tools

from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep

import paths

# Module-level constants
USER = os.environ["USER"]
SCRATCH = paths.scratch

CACHE_DIR = SCRATCH / "atlas_cache"
S3_BASE_URL = "s3://us-west-2.opendata.source.coop/cworthy/oae-efficiency-atlas/data"

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Initialize cache directory
os.makedirs(CACHE_DIR, exist_ok=True)


class DatasetSpec:
    def __init__(
        self,
        name,
        s3_base_url,
        n_years,
        polygon_ids,
        injection_years,
        injection_months,
        model_year_align: int = 0,
        model_year_offset: int = 0,
    ):
        self.name = name
        self.s3_base_url = s3_base_url
        self.n_years = n_years
        self.polygon_ids = polygon_ids
        self.injection_years = injection_years
        self.injection_months = injection_months
        self.model_year_align = model_year_align
        self.model_year_offset = model_year_offset

    @property
    def df(self):
        return self._df()

    def _build_year_month_pairs(self, injection_month):
        year_month_pairs = []
        month = injection_month
        year = self.model_year_align - self.model_year_offset
        for n in range(self.n_years * 12):
            year_month_pairs.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return year_month_pairs        

    def get_polygon_masks_dataset(self, force_download: bool = False) -> xr.Dataset:
        """
        Return the polygon masks dataset, downloading and caching if needed.

        Parameters
        ----------
        force_download : bool, optional
            If True, re-download even if cached. Default is False.
        """
        polygon_masks_url = f"{self.s3_base_url}/polygon_masks.nc"
        polygon_masks_cache_path = CACHE_DIR / "polygon_masks.nc"
        if force_download or not polygon_masks_cache_path.exists():
            polygon_masks_cache_path.parent.mkdir(parents=True, exist_ok=True)
            fs = s3fs.S3FileSystem(anon=True)
            fs.get(polygon_masks_url, str(polygon_masks_cache_path))
        return xr.open_dataset(polygon_masks_cache_path)

    def _df(self):
        """Build a DataFrame of all expected S3 and cache paths."""
        records = []
        for injection_year in self.injection_years:
            for injection_month in self.injection_months:
                for polygon_id in self.polygon_ids:
                    for year, month in self._build_year_month_pairs(injection_month):
                        records.append(
                            {
                                "injection_year": injection_year,
                                "polygon_id": polygon_id,
                                "injection_month": injection_month,
                                "year": year,
                                "month": month,
                                "s3_path": (
                                    f"{self.s3_base_url}/experiments/{polygon_id:03d}/{injection_month:02d}/"
                                    f"alk-forcing.{polygon_id:03d}-{injection_year:04d}-{injection_month:02d}"
                                    f".pop.h.{year:04d}-{month:02d}.nc"
                                ),
                                "cache_path": (
                                    f"{CACHE_DIR}/experiments/{polygon_id:03d}/{injection_month:02d}/"
                                    f"alk-forcing.{polygon_id:03d}-{injection_year:04d}-{injection_month:02d}"
                                    f".pop.h.{year:04d}-{month:02d}.nc"
                                ),
                            }
                        )
        df = pd.DataFrame.from_records(records)
        df.set_index(
            ["injection_year", "polygon_id", "injection_month", "year", "month"],
            inplace=True,
        )
        return df

    def query(self, polygon_id=None, injection_year=None, injection_month=None, n_test=None):

        manifest_df = self.df.reset_index()
        if polygon_id is not None:
            manifest_df = manifest_df[manifest_df["polygon_id"] == polygon_id]
        if injection_year is not None:
            manifest_df = manifest_df[manifest_df["injection_year"] == injection_year]
        if injection_month is not None:
            manifest_df = manifest_df[manifest_df["injection_month"] == injection_month]
        if n_test is not None:
            manifest_df = manifest_df.head(n_test)
        
        return manifest_df

    def ensure_cache(
        self,
        polygon_id=None,
        injection_year=None,
        injection_month=None,
        n_test=None,
        batch_size=50,
    ):
        
        manifest_df = self.query(
            polygon_id=polygon_id,
            injection_year=injection_year,
            injection_month=injection_month,
            n_test=n_test,
        )

        manifest = (
            manifest_df[["s3_path", "cache_path"]]
            .to_dict(orient="records")
        )   

        s3_files = [entry["s3_path"] for entry in manifest]
        cache_paths = [Path(entry["cache_path"]) for entry in manifest]
        n_downloaded = _download_missing_files(s3_files, cache_paths, batch_size=batch_size)
        return manifest

    def open_dataset(self, polygon_id=None, injection_year=None, injection_month=None, n_test=None):
        manifest = self.ensure_cache(
            polygon_id=polygon_id,
            injection_year=injection_year,
            injection_month=injection_month,
            n_test=n_test,
        )
        cache_paths = [entry["cache_path"] for entry in manifest]
        return xr.open_mfdataset(
            cache_paths,
            combine="by_coords",
            decode_timedelta=True,
        )


DATASET_REGISTRY = {
    "oae-efficiency-map_atlas-v0": DatasetSpec(
        name="oae-efficiency-map_atlas-v0",
        s3_base_url=S3_BASE_URL,
        n_years=15,
        polygon_ids=list(range(0, 690)),
        injection_years=[1999],
        injection_months=[1, 4, 7, 10],
        model_year_align=1999,
        model_year_offset=1652,
    )
}



def get_pop_grid() -> xr.Dataset:
    """
    Return the POP grid dataset, downloading and caching if needed.
    """
    return pop_tools.get_grid(grid_name="POP_gx1v7")
    


def points_within_grid_boundaries(model_grid, points_lat, points_lon):
    """
    Return indices of points that are inside the model grid boundaries.
    
    Uses the convex hull of model_grid.ds.lat_u and lon_u to determine boundaries
    via a point-in-polygon test.
    
    Parameters
    ----------
    model_grid : Grid
        Model grid object with ds attribute containing lat_u and lon_u.
    points_lat : array-like
        Array of latitude values.
    points_lon : array-like
        Array of longitude values.
    
    Returns
    -------
    numpy.ndarray
        Array of indices where points are within the model grid boundaries.
    """
    # Extract all lat/lon points from the model grid
    grid_lat = model_grid.ds.lat_u.values
    grid_lon = model_grid.ds.lon_u.values
    
    # Flatten to 1D arrays
    grid_lat_flat = grid_lat.ravel()
    grid_lon_flat = grid_lon.ravel()
    
    # Remove any NaN or invalid values
    valid_mask = np.isfinite(grid_lat_flat) & np.isfinite(grid_lon_flat)
    grid_lat_flat = grid_lat_flat[valid_mask]
    grid_lon_flat = grid_lon_flat[valid_mask]
    
    # Normalize longitudes to avoid dateline/0-degree wrap issues.
    def _unwrap_longitudes(lons, reference):
        lons = np.asarray(lons, dtype=float)
        reference = float(reference)
        return ((lons - reference + 180.0) % 360.0) - 180.0 + reference

    lon_reference = np.nanmedian(grid_lon_flat)
    grid_lon_flat = _unwrap_longitudes(grid_lon_flat, lon_reference)

    # Create MultiPoint from grid points (shapely uses lon, lat order)
    grid_points = MultiPoint(list(zip(grid_lon_flat, grid_lat_flat)))
    
    # Compute convex hull and prepare it for faster contains checks
    convex_hull = grid_points.convex_hull
    prepared_hull = prep(convex_hull)
    
    # Convert input points to numpy arrays
    points_lat = np.asarray(points_lat)
    points_lon = np.asarray(points_lon)
    points_lon = _unwrap_longitudes(points_lon, lon_reference)
    
    # Test each point for containment in the convex hull
    # Use prepared geometry for faster contains checks
    within_bounds = np.zeros(len(points_lat), dtype=bool)
    for i in range(len(points_lat)):
        if np.isfinite(points_lat[i]) and np.isfinite(points_lon[i]):
            point = Point(points_lon[i], points_lat[i])
            # Check if point is within or on the boundary of the convex hull
            within_bounds[i] = prepared_hull.contains(point) or convex_hull.touches(point)
    
    # Return indices of points within boundaries
    return np.where(within_bounds)[0]


def _download_missing_files(s3_files, cache_paths, force_download=False, batch_size=50):
    """
    Download missing files from S3 in batches.
    """
    to_download = []
    for s3_file, cache_path in zip(s3_files, cache_paths):
        if force_download or not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            to_download.append((s3_file, cache_path))

    if not to_download:
        return 0

    fs = s3fs.S3FileSystem(anon=True)
    for start in range(0, len(to_download), batch_size):
        chunk = to_download[start : start + batch_size]
        if not chunk:
            continue
        s3_chunk = [s3 for s3, _ in chunk]
        path_chunk = [str(path) for _, path in chunk]
        if len(s3_chunk) != len(path_chunk):
            raise ValueError("S3 and cache path lists are mismatched.")
        try:
            fs.get(s3_chunk, path_chunk)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {len(s3_chunk)} file(s) from S3."
            ) from exc
    return len(to_download)


class AtlasModelGridAnalyzer:
    """
    Analyzer for computing properties of atlas dataset subset to model grid.
    
    The atlas dataset has dimensions (polygon_id, injection_date, elapsed_time, nlat, nlon).
    This class provides methods for various dimension reduction operations including
    identifying polygons within grid boundaries and integrating FG_CO2 data.
    
    Attributes
    ----------
    model_grid : Grid
        Model grid object with ds attribute containing lat_u and lon_u.
    """
    
    def __init__(self, model_grid, atlas_data):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        model_grid : Grid
            Model grid object with ds attribute containing lat_u and lon_u. 
        atlas_data : DatasetSpec
            Atlas data object with polygon_id, injection_year, injection_month, and years.
        """
        self.model_grid = model_grid
        self.atlas_data = atlas_data
        self.atlas_grid = get_pop_grid()
        self._polygon_mask = None
        
        # Cache for computed properties
        self._within_boundaries_indices = None
        self.polygon_ids_in_bounds = None
        
        # Compute polygon IDs within boundaries on initialization
        self._get_polygon_ids_within_boundaries()
    
    @property
    def polygon_mask(self):
        if self._polygon_mask is None:
            ds_atlas_polygons = self.atlas_data.get_polygon_masks_dataset()
            self._polygon_mask = ds_atlas_polygons.polygon_id.where(self.atlas_grid.KMT > 0)
        return self._polygon_mask

    def _get_polygon_ids_within_boundaries(self):
        """
        Identify polygon IDs that are within the model grid boundary.
        
        Uses points_within_grid_boundaries to find indices, then indexes
        the polygon_id field to get unique polygon IDs.
        
        Returns
        -------
        numpy.ndarray
            Unique array of polygon IDs within the model grid boundaries.
        """
        
        # Get indices of points within boundaries
        indices = points_within_grid_boundaries(
            self.model_grid,
            self.atlas_grid.TLAT.values.ravel(),
            self.atlas_grid.TLONG.values.ravel()
        )
        self._within_boundaries_indices = indices
        
        # Get polygon IDs at those indices
        polygon_id_flat = self.polygon_mask.values.ravel()
        polygon_ids_in_bounds = polygon_id_flat[indices]
        
        # Get unique polygon IDs (exclude negative values which may indicate invalid/masked)
        self.polygon_ids_in_bounds = np.unique(
            polygon_ids_in_bounds[polygon_ids_in_bounds >= 0]
        )
        
        self.polygon_id_mask = self.polygon_mask.copy()
        nlat, nlon = self.polygon_id_mask.shape
        within_mask_flat = np.zeros(nlat * nlon, dtype=bool)
        within_mask_flat[self._within_boundaries_indices] = True
        within_mask = within_mask_flat.reshape(nlat, nlon)
        self.polygon_id_mask = self.polygon_id_mask.where(within_mask, -1).where(self.atlas_grid.KMT > 0)

    def set_field_within_boundaries(self, field_by_id):
        """
        Set fields within boundaries.
        """
        
        nlat, nlon = self.polygon_id_mask.shape
        within_mask_flat = np.zeros(nlat * nlon, dtype=bool)
        within_mask_flat[self._within_boundaries_indices] = True
        within_mask = within_mask_flat.reshape(nlat, nlon)

        field_flat = self.polygon_id_mask.copy().where(within_mask, 0.0).where(self.atlas_grid.KMT > 0).values.ravel()
        field_flat[:] = np.nan
        for n, id in enumerate(field_by_id.polygon_id.values):
            ndx = np.where(self.polygon_id_mask.values.ravel() == id)[0]
            field_flat[ndx] = field_by_id[n]

        field_mapped = self.polygon_id_mask.copy()
        field_mapped.values = field_flat.reshape(nlat, nlon)        
        return field_mapped

    def integrate_fg_co2_polygon_by_id(
        self, polygon_id, injection_year, injection_month, years=None, n_test=None
    ):
        """
        Integrate FG_CO2 over time, lat, and lon using TAREA for area weighting.
        
        Restricts computation to lat/lon area within the model grid boundaries.
        Returns cumulative integrals over elapsed_time.
        
        Parameters
        ----------
        polygon_id : int or float
            Polygon ID. Will be converted to integer.
        injection_year : int
            Injection year (e.g., 1999).
        injection_month : int
            Injection month (1-12).
        years : list of int, optional
            List of year integers (e.g., [347, 348, 349]).
        n_test : int, optional
            If provided, limit the download to the first n_test files.
        
        Returns
        -------
        xarray.Dataset
            Dataset with data variables:
            - 'total': Cumulative integrated FG_CO2 over all space (elapsed_time dimension)
            - 'within_grid': Cumulative integrated FG_CO2 within model grid boundaries 
              (elapsed_time dimension)
            - 'fraction': Fraction of uptake within model grid (elapsed_time dimension)
        
        Raises
        ------
        ValueError
            If n_test is provided but is not a positive integer.
        KeyError
            If FG_CO2 variable is not found in dataset.
        """
               
        # Print plan before starting
        injection_date = f"{injection_year:04d}-{injection_month:02d}"
        
        # Read alk-forcing data
        ds = self.atlas_data.open_dataset(
            polygon_id=polygon_id,
            injection_year=injection_year,
            injection_month=injection_month,
            n_test=n_test
        )
        try:
            if 'FG_CO2' not in ds:
                raise KeyError("FG_CO2 variable not found in dataset")
            
            if 'FG_ALT_CO2' not in ds:
                raise KeyError("FG_ALT_CO2 variable not found in dataset")

            year_month_pairs = self.atlas_data._build_year_month_pairs(injection_month)
            if n_test is not None:
                year_month_pairs = year_month_pairs[:n_test]
            if len(year_month_pairs) > len(ds.elapsed_time):
                year_month_pairs = year_month_pairs[:len(ds.elapsed_time)]

            # Create time_delta DataArray with days per month, following the year/month pairs
            time_delta_seconds = xr.DataArray(
                [DAYS_PER_MONTH[month - 1] * 86400.0 for _, month in year_month_pairs],
                dims=['elapsed_time'],
                coords={'elapsed_time': ds.elapsed_time} if 'elapsed_time' in ds.coords else None
            )
            
            # Get TAREA for area weighting
            tarea = self.atlas_grid.TAREA
            nlat, nlon = tarea.shape

            # Get indices within boundaries if not already computed
            if self._within_boundaries_indices is None:
                self._get_polygon_ids_within_boundaries()
            
            # Create mask for points within grid boundaries
            within_mask = np.zeros((nlat, nlon), dtype=bool)
            within_mask.ravel()[self._within_boundaries_indices] = True

            # Select and process FG_CO2 data
            fg_co2_alt_co2 = ds.FG_ALT_CO2.sel(
                polygon_id=polygon_id, injection_date=injection_date
            ).squeeze()

            fg_co2 = ds.FG_CO2.sel(
                polygon_id=polygon_id, injection_date=injection_date
            ).squeeze()
            

            fg_co2_additional = fg_co2 - fg_co2_alt_co2 # nmol/cm2/s
            fg_co2_additional *= 1e-9 # mol/cm2/s

            # Compute cumulative integrals
            fg_co2_int_within = (
                (fg_co2_additional * time_delta_seconds * tarea.where(within_mask))
                .sum(dim=['nlat', 'nlon'])
                .cumsum(dim='elapsed_time')
            )
            fg_co2_int_total = (
                (fg_co2_additional * time_delta_seconds * tarea)
                .sum(dim=['nlat', 'nlon'])
                .cumsum(dim='elapsed_time')
            )
            fraction = fg_co2_int_within / fg_co2_int_total
            
            # Return as xarray Dataset
            results = xr.Dataset({
                'total': fg_co2_int_total,
                'within_grid': fg_co2_int_within,
                'fraction': fraction
            }).compute()
            return results
        finally:
            ds.close()
    
    def integrate_fg_co2_polygons_within_boundaries(
        self, injection_year=None, injection_month=None, n_test=None
    ):
        """
        Integrate FG_CO2 for all polygons within model grid boundaries.
        
        Computes integrals for all polygons within boundaries and concatenates
        results along the polygon_id dimension.
        
        Parameters
        ----------
        injection_year : int
            Injection year (e.g., 1999).
        injection_month : int
            Injection month (1-12).
        n_test : int, optional
            If provided, limit the download to the first n_test files.
        
        Returns
        -------
        xarray.Dataset
            Concatenated dataset with polygon_id dimension containing results 
            for all polygons. Has dimensions (polygon_id, elapsed_time).
        """
        datasets = []
        for n, polygon_id in enumerate(self.polygon_ids_in_bounds):
            result = self.integrate_fg_co2_polygon_by_id(
                polygon_id=polygon_id,
                injection_year=injection_year,
                injection_month=injection_month,
                n_test=n_test
            )
            # Add polygon_id as a coordinate to each dataset
            result = result.assign_coords(polygon_id=polygon_id)
            datasets.append(result)

            if n_test is not None and n >= 1:
                break

        # Concatenate along polygon_id dimension
        return xr.concat(datasets, dim='polygon_id')

