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

# Third-party imports   
import numpy as np
import s3fs
import xarray as xr
from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep

# Module-level constants
CACHE_DIR = "atlas_cache"

CACHE_DIR_ROOT = Path(os.environ.get("SCRATCH", Path(os.environ.get("HOME")) / "scratch"))
CACHE_DIR = CACHE_DIR_ROOT / "atlas_cache"

S3_BASE_URL = "s3://us-west-2.opendata.source.coop/cworthy/oae-efficiency-atlas/data"
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Initialize cache directory
os.makedirs(CACHE_DIR, exist_ok=True)


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
    
    # Create MultiPoint from grid points (shapely uses lon, lat order)
    grid_points = MultiPoint(list(zip(grid_lon_flat, grid_lat_flat)))
    
    # Compute convex hull and prepare it for faster contains checks
    convex_hull = grid_points.convex_hull
    prepared_hull = prep(convex_hull)
    
    # Convert input points to numpy arrays
    points_lat = np.asarray(points_lat)
    points_lon = np.asarray(points_lon)
    
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


def get_atlas_data(polygon_id, injection_year, injection_month, year, month, force_download=False):
    """
    Retrieve atlas data from S3 for a given polygon, injection date, and time period.
    
    Uses local cache to avoid re-downloading files. Files are cached in a directory
    structure mirroring the S3 path.
    
    Parameters
    ----------
    polygon_id : int or float
        Polygon ID (e.g., 0, 1, 2). Will be converted to integer.
    injection_year : int or float
        Injection year (e.g., 1999). Will be converted to integer.
    injection_month : int or float
        Injection month (1-12). Will be converted to integer.
    year : int or float
        Year of the data file (e.g., 347). Will be converted to integer.
    month : int or float
        Month of the data file (1-12). Will be converted to integer.
    force_download : bool, optional
        If True, force download even if file exists in cache. Default is False.
    
    Returns
    -------
    pathlib.Path
        Path to the local cached file.
    
    Raises
    ------
    AssertionError
        If exactly one matching file is not found in S3.
    """
    # Convert to integers to ensure proper formatting
    polygon_id = int(polygon_id)
    injection_year = int(injection_year)
    injection_month = int(injection_month)
    year = int(year)
    month = int(month)
    
    fs = s3fs.S3FileSystem(anon=True)
    
    # Construct S3 path
    s3_prefix = f"{S3_BASE_URL}/experiments/{polygon_id:03d}/{injection_month:02d}/"
    all_files = fs.ls(s3_prefix)
    
    # Filter files to include only those ending in {year:04d}-{month:02d}.nc
    pattern = f"{year:04d}-{month:02d}.nc"
    s3_files = [f for f in all_files if f.endswith(pattern)]
    
    if len(s3_files) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching pattern {pattern} in {s3_prefix}, "
            f"found {len(s3_files)}"
        )
    
    s3_file = s3_files[0]
    
    # Construct local cache path
    cache_path = (
        Path(CACHE_DIR) 
        / f"experiments/{polygon_id:03d}/{injection_month:02d}" 
        / os.path.basename(s3_file)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists locally and download if needed
    if force_download or not cache_path.exists():
        fs.get(s3_file, str(cache_path))
    
    return cache_path


def read_alk_forcing_files(polygon_id, injection_year, injection_month, years, months):
    """
    Read and combine alk-forcing files for a given polygon ID and time period.
    
    Parameters
    ----------
    polygon_id : int or float
        Polygon ID (e.g., 0, 1, 2). Will be converted to integer.
    injection_year : int
        Injection year (e.g., 1999).
    injection_month : int
        Injection month (1-12).
    years : list of int
        List of year integers (e.g., [347, 348, 349]).
    months : list of int
        List of month numbers (1-12). Must be provided.
    
    Returns
    -------
    xarray.Dataset
        Combined dataset with all requested time periods concatenated along time dimension.
    
    Raises
    ------
    ValueError
        If months is None or empty.
    FileNotFoundError
        If any requested file is not found.
    """
    if months is None or len(months) == 0:
        raise ValueError("months must be provided as a non-empty list")
    
    files = []
    for year in years:
        for month in months:
            filepath = get_atlas_data(
                polygon_id, injection_year, injection_month, year, month, 
                force_download=False
            )
            if not filepath.exists():
                raise FileNotFoundError(
                    f"File not found after download attempt: {filepath}"
                )
            files.append(str(filepath))
    
    # Open and combine datasets
    return xr.open_mfdataset(files, decode_timedelta=True)


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
    atlas_grid : xarray.Dataset
        Atlas grid dataset with TLAT, TLONG, and TAREA variables.
    polygon_ids : xarray.DataArray
        DataArray containing polygon_id field.
    polygon_ids_in_bounds : numpy.ndarray
        Unique array of polygon IDs within the model grid boundaries.
    """
    
    def __init__(self, model_grid, atlas_grid, polygon_ids):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        model_grid : Grid
            Model grid object with ds attribute containing lat_u and lon_u.
        atlas_grid : xarray.Dataset
            Atlas grid dataset with TLAT, TLONG, and TAREA variables.
        polygon_ids : xarray.DataArray
            DataArray containing polygon_id field.
        """
        self.model_grid = model_grid
        self.atlas_grid = atlas_grid
        self.polygon_ids = polygon_ids
        
        # Cache for computed properties
        self._within_boundaries_indices = None
        self.polygon_ids_in_bounds = None
        
        # Compute polygon IDs within boundaries on initialization
        self._get_polygon_ids_within_boundaries()
    
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
        polygon_id_flat = self.polygon_ids.values.ravel()
        polygon_ids_in_bounds = polygon_id_flat[indices]
        
        # Get unique polygon IDs (exclude negative values which may indicate invalid/masked)
        self.polygon_ids_in_bounds = np.unique(
            polygon_ids_in_bounds[polygon_ids_in_bounds >= 0]
        )
        
        self.polygon_id_mask = self.polygon_ids.copy()
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
        for n, id in enumerate(field_by_id.polygon_id.values):
            ndx = np.where(self.polygon_id_mask.values.ravel() == id)[0]
            field_flat[ndx] = field_by_id[n]

        field_mapped = self.polygon_id_mask.copy()
        field_mapped.values = field_flat.reshape(nlat, nlon)
        return field_mapped

    def integrate_fg_co2_polygon(
        self, polygon_id, years, months=None, injection_year=1999, injection_month=1
    ):
        """
        Integrate FG_CO2 over time, lat, and lon using TAREA for area weighting.
        
        Restricts computation to lat/lon area within the model grid boundaries.
        Returns cumulative integrals over elapsed_time.
        
        Parameters
        ----------
        polygon_id : int or float
            Polygon ID. Will be converted to integer.
        years : list of int
            List of year integers (e.g., [347, 348, 349]).
        months : list of int, optional
            List of month numbers (1-12). If None, uses all months (1-12).
            Default is None.
        injection_year : int, optional
            Injection year. Default is 1999.
        injection_month : int, optional
            Injection month (1-12). Default is 1.
        
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
            If months is not a valid list of integers.
        KeyError
            If FG_CO2 variable is not found in dataset.
        """
        # Validate and set default months
        if months is None:
            months = list(range(1, 13))
        else:
            if not isinstance(months, list):
                raise ValueError("months must be a list of integers")
            if not all(isinstance(m, int) for m in months):
                raise ValueError("months must be a list of integers")
            if not all(1 <= m <= 12 for m in months):
                raise ValueError("months must be integers between 1 and 12")
        
        # Read alk-forcing data
        ds = read_alk_forcing_files(
            polygon_id, injection_year, injection_month, years, months
        )

        # Get FG_CO2 data
        if 'FG_CO2' not in ds:
            raise KeyError("FG_CO2 variable not found in dataset")
        
        if 'FG_ALT_CO2' not in ds:
            raise KeyError("FG_ALT_CO2 variable not found in dataset")

        # Create time_delta DataArray with days per month, replicating months for each year
        # For each year, repeat the months list
        time_delta_seconds = xr.DataArray(
            [DAYS_PER_MONTH[m - 1] * 86400.0 for _ in years for m in months],
            dims=['elapsed_time'],
            coords={'elapsed_time': ds.elapsed_time} if 'elapsed_time' in ds.coords else None
        )
        
        # Get TAREA for area weighting
        tarea = self.atlas_grid.TAREA
        nlat, nlon = tarea.shape

        # Select the appropriate polygon and injection date
        injection_date = f"{injection_year:04d}-{injection_month:02d}"

        # Get indices within boundaries if not already computed
        if self._within_boundaries_indices is None:
            self._get_polygon_ids_within_boundaries()
        
        # Create mask for points within grid boundaries
        within_mask = np.zeros((nlat, nlon), dtype=bool)
        within_mask.ravel()[self._within_boundaries_indices] = True
        

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
        return xr.Dataset({
            'total': fg_co2_int_total,
            'within_grid': fg_co2_int_within,
            'fraction': fraction
        })
    
    def integrate_fg_co2_all_polygons(
        self, years, months=None, injection_year=1999, injection_month=1
    ):
        """
        Integrate FG_CO2 for all polygons within model grid boundaries.
        
        Computes integrals for all polygons within boundaries and concatenates
        results along the polygon_id dimension.
        
        Parameters
        ----------
        years : list of int
            List of year integers (e.g., [347, 348, 349]).
        months : list of int, optional
            List of month numbers (1-12). If None, uses all months (1-12).
            Default is None.
        injection_year : int, optional
            Injection year. Default is 1999.
        injection_month : int, optional
            Injection month (1-12). Default is 1.
        
        Returns
        -------
        xarray.Dataset
            Concatenated dataset with polygon_id dimension containing results 
            for all polygons. Has dimensions (polygon_id, elapsed_time).
        """
        datasets = []
        for polygon_id in self.polygon_ids_in_bounds:
            result = self.integrate_fg_co2_polygon(
                polygon_id, years, months, injection_year, injection_month
            )
            # Add polygon_id as a coordinate to each dataset
            result = result.assign_coords(polygon_id=polygon_id)
            datasets.append(result)
        # Concatenate along polygon_id dimension
        return xr.concat(datasets, dim='polygon_id')
