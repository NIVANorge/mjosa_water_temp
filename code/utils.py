import datetime as dt
import warnings

import geopandas as gpd
import nivapy3 as nivapy
import pandas as pd


def get_nve_gts_api_aggregated_time_series(
    poly_gdf,
    pars,
    st_dt,
    end_dt,
    id_col="station_code",
    n_samp=None,
    random_state=None,
):
    """Get time series for the parameters and time period of interest, aggregated over the
    polygons in 'poly_gdf'. Data comes from NVE's GridTimeSeries API
    (http://api.nve.no/doc/gridtimeseries-data-gts/).

    Args
        poly_gdf: Geodataframe. Polygons of interest. Make sure the CRS is set and valid.
        pars: Dataframe or list. If dataframe, must be in the format returned by
            get_nve_gts_api_parameters(), filtered to the parameters of interest. If list,
            must be a list of 'str' matching valid parameter names in the 'Name' column
            returned by get_nve_gts_api_parameters()
        st_dt: Str. Start date of interest 'YYYY-MM-DD'
        end_dt: Str. End date of interest 'YYYY-MM-DD'
        id_col: Str. Name of column in 'poly_gdf' containing a unique ID for each polygon
            of interest
        n_samp: Int or None. Number of points to sample per polygon. If None, get data for
            all grid cells within the polygon and calculate summary statistics based on the
            full dataset. Requires one API call per grid cell, which can be slow for large
            catchments. If Int and the number of 1 km2 grid cells within the polygon is
            larger than the value specified, 'n_samp' grid cells within the polygon are
            selected at random and summary statistics are based on the sample, instead of
            all values. If the total number of cells is less than or equal to 'n_samp', all
            points will be used (i.e. same behaviour as 'None').
        random_state: Int or None. Used for random sampling when 'n_samp' is not None.

    Returns
        Dataframe of aggregated time series data for each polygon.
    """
    # Validate user input
    if len(poly_gdf[id_col].unique()) != len(poly_gdf):
        raise ValueError("ERROR: 'id_col' is not unique.")

    if dt.datetime.strptime(st_dt, "%Y-%m-%d") > dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ):
        raise ValueError("'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD').")

    par_df = nivapy.da.get_nve_gts_api_parameters()
    if isinstance(pars, pd.DataFrame):
        pars = list(pars["Name"])
    assert set(pars).issubset(
        list(par_df["Name"])
    ), "Some parameters in 'pars' not recognised."

    # Reproject to CRS required by API (EPSG 25833)
    poly_gdf = poly_gdf.copy().to_crs("epsg:25833")

    # Build gdf of points at grid cell centres on a 1 km grid
    # Norway bounding box in EPSG 25833
    xmin, ymin, xmax, ymax = -80000, 6449000, 1120000, 7945000
    pt_df = nivapy.da.create_point_grid(xmin, ymin, xmax, ymax, 1000)
    pt_gdf = gpd.GeoDataFrame(
        pt_df,
        geometry=gpd.points_from_xy(pt_df["x"], pt_df["y"], crs="epsg:25833"),
    )

    # Get just points within polys
    pt_gdf = gpd.sjoin(pt_gdf, poly_gdf, predicate="intersects", how="inner")
    pt_df = pd.DataFrame(pt_gdf[[id_col, "point_id", "x", "y"]])

    # Select 'n_samp' points per site, if desired
    if n_samp:
        pt_df = pt_df.groupby(id_col, group_keys=False).apply(
            lambda g: (
                g
                if len(g) <= n_samp
                else g.sample(n_samp, replace=False, random_state=random_state)
            )
        )

    # Get data for points from API
    res_df = nivapy.da.get_nve_gts_api_time_series(
        pt_df,
        pars,
        st_dt,
        end_dt,
        id_col="point_id",
        xcol="x",
        ycol="y",
        crs="epsg:25833",
    )

    # Join poly IDs
    res_df = pd.merge(res_df, pt_df[["point_id", id_col]], how="left", on="point_id")

    # Aggregate
    res_df = (
        res_df.groupby([id_col, "par", "datetime"])
        .agg(
            {
                "altitude_m": ["mean"],
                "full_name": "first",
                "unit": "first",
                "time_resolution": "first",
                "value": ["min", "median", "max", "mean", "std", "count"],
            }
        )
        .reset_index()
    )

    res_df.columns = ["_".join(i) for i in res_df.columns.to_flat_index()]
    res_df.rename(
        {
            f"{id_col}_": id_col,
            "par_": "par",
            "datetime_": "datetime",
            "altitude_m_mean": "mean_altitude_m",
            "full_name_first": "full_name",
            "unit_first": "unit",
            "time_resolution_first": "time_resolution",
        },
        axis="columns",
        inplace=True,
    )

    if len(res_df[id_col].unique()) < len(poly_gdf[id_col].unique()):
        missing = sorted(
            list(set(poly_gdf[id_col].unique()) - set(res_df[id_col].unique()))
        )
        msg = (
            "The following catchments do not contain any grid cell centres:\n"
            f"{missing}\n"
            "Summary statistics for these catchments have not been calculated. "
            "This is a known limitation that will be fixed soon."
        )
        warnings.warn(msg)

    return res_df