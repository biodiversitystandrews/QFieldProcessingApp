import glob
import os
import re
from datetime import datetime

import fiona
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

# Example data:
# geom (hidden), fid (auto), year, month, day, species, date, observer, height, radius, photoid, count, year1, comment, type, english_name, taxa
# Example row: Point (-2.79425316963482118 56.33810287940671913) 1 2024-25 02 08 Calluna vulgaris 2025-02-08 Alice Mortimer 2025 Angiosperm Heather Angiosperm

# Standard output columns. Alternative names are accepted from older or slightly different GPKG exports.
STANDARD_COLUMNS = [
    {"name": "geom", "datatype": "geometry", "alt_names": ["geom", "geometry"]},
    {"name": "species", "datatype": "text", "alt_names": ["species", "species name", "scientific_name", "scientific name"]},
    {"name": "english_name", "datatype": "text", "alt_names": ["english_name", "english name", "english"]},
    {"name": "Date", "datatype": "date", "alt_names": ["date", "date_obs", "date observed", "date_observed"]},
    {"name": "year", "datatype": "text", "alt_names": ["school_year", "year"]},
    {"name": "year1", "datatype": "numeric", "alt_names": ["calendar_year", "year1"]},
    {"name": "month", "datatype": "numeric", "alt_names": ["month"]},
    {"name": "day", "datatype": "numeric", "alt_names": ["day"]},
    {"name": "Taxa", "datatype": "text", "alt_names": ["taxa"]},
    {"name": "obs", "datatype": "text", "alt_names": ["observer", "observer name", "observer_name", "obs"]},
    {"name": "comment", "datatype": "text", "alt_names": ["comment", "comments"]},
    {"name": "height", "datatype": "numeric", "alt_names": ["height"]},
    {"name": "radius", "datatype": "numeric", "alt_names": ["radius"]},
    {"name": "photoid", "datatype": "text", "alt_names": ["photoid", "photo_id"]},
    {"name": "count", "datatype": "numeric", "alt_names": ["count"]},
]

# Values that should be treated as missing text, not preserved as literal words in output columns.
TEXT_NULLS = {"", "nan", "none", "null", "n/a", "na"}

# Standard geometry and date column names, pulled from STANDARD_COLUMNS to avoid hardcoding them repeatedly.
GEOM_COL = next(col["name"] for col in STANDARD_COLUMNS if col["datatype"] == "geometry")
DATE_COL = next(col["name"] for col in STANDARD_COLUMNS if col["datatype"] == "date")

# Final column order used when writing standardised GeoDataFrames.
STANDARD_ORDER = [col["name"] for col in STANDARD_COLUMNS]

# Lookup table from every accepted column-name variant to its standard output column name.
COLUMN_NAME_MAP = {
    re.sub(r"_+", "_", str(alt).strip().lower().replace(" ", "_")): col["name"]
    for col in STANDARD_COLUMNS
    for alt in [col["name"], *col["alt_names"]]
}


# Converts names into a consistent format, e.g. "Observer Name" -> "observer_name".
def clean_column_key(name):
    return re.sub(r"_+", "_", str(name).strip().lower().replace(" ", "_"))


# Normalises species names so matching is case and punctuation insensitive.
def normalise_species_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", str(value).lower().strip())


# Cleans text fields without turning missing values into the strings "None" or "nan".
def clean_text_series(series, title_case=True):
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.mask(cleaned.str.lower().isin(TEXT_NULLS), pd.NA)
    return cleaned.str.lower().str.title() if title_case else cleaned


# Converts numeric fields stored as text into numbers, coercing invalid values to missing.
def clean_numeric_series(series):
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


# Merges duplicate column names by keeping the first non-empty value across duplicates.
def merge_duplicate_columns(gdf):
    result = pd.DataFrame(index=gdf.index)
    for col in dict.fromkeys(gdf.columns):
        matches = gdf.loc[:, gdf.columns == col]
        result[col] = matches.iloc[:, 0] if matches.shape[1] == 1 else matches.bfill(axis=1).iloc[:, 0]

    geometry_name = getattr(getattr(gdf, "geometry", None), "name", None)
    if geometry_name in result.columns:
        return gpd.GeoDataFrame(result, geometry=geometry_name, crs=getattr(gdf, "crs", None))
    return result


# Cleans incoming column names and maps accepted alternatives onto standard column names.
def normalise_input_columns(gdf):
    gdf = gdf.copy()
    gdf.columns = [clean_column_key(col) for col in gdf.columns]
    gdf = merge_duplicate_columns(gdf)

    for source in list(gdf.columns):
        target = COLUMN_NAME_MAP.get(source)
        if target is None or source == target:
            continue
        if target in gdf.columns:
            gdf[target] = gdf[target].combine_first(gdf[source])
            gdf = gdf.drop(columns=[source])
        else:
            gdf = gdf.rename(columns={source: target})
    return gdf


# Converts point geometry values or WKT point strings into Shapely Points; invalid values become missing.
def parse_geometry(value):
    if isinstance(value, Point):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = wkt.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, Point) else None
    return None


# Converts one date value into a timezone-free pandas timestamp.
def standardise_date_value(value):
    if pd.isna(value):
        return pd.NaT
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


# Converts a whole date column into one consistent timezone-free datetime dtype.
def standardise_date_series(series):
    return pd.to_datetime(series.apply(standardise_date_value), errors="coerce").astype("datetime64[ms]")


# Creates an empty GeoDataFrame with all standard columns in the expected order.
def empty_standard_gdf(crs="EPSG:4326"):
    data = {
        col["name"]: pd.Series(dtype="object")
        for col in STANDARD_COLUMNS
        if col["datatype"] != "geometry"
    }
    data[GEOM_COL] = gpd.GeoSeries([], crs=crs)
    return gpd.GeoDataFrame(data, geometry=GEOM_COL, crs=crs)


# Adds any missing standard columns so the output schema is always complete.
def add_missing_standard_columns(gdf):
    for column in STANDARD_COLUMNS:
        if column["name"] not in gdf.columns:
            gdf[column["name"]] = None
    return gdf


# Applies standard text and numeric cleaning to fields that are present.
def clean_standard_fields(gdf):
    for column in STANDARD_COLUMNS:
        name = column["name"]
        if name not in gdf.columns:
            continue
        if column["datatype"] == "text" and name != "species":
            gdf[name] = clean_text_series(gdf[name], title_case=True)
        elif column["datatype"] == "numeric":
            gdf[name] = clean_numeric_series(gdf[name])

    if "species" in gdf.columns:
        gdf["species"] = clean_text_series(gdf["species"], title_case=False)
    return gdf


# Drops rows missing the minimum fields needed for a valid observation.
def drop_missing_critical_data(gdf, label):
    critical_cols = ["geom", "species", "Date", "obs"]
    existing_critical_cols = [col for col in critical_cols if col in gdf.columns]
    if not existing_critical_cols:
        return gdf

    before = len(gdf)
    gdf = gdf.dropna(subset=existing_critical_cols)
    print(f"Dropped {before - len(gdf)} records from {label} with missing critical data: {', '.join(existing_critical_cols)}")
    return gdf


# Standardises a GeoDataFrame to the expected pipeline format.
def standardise(gdf, label="gdf"):
    if gdf.empty:
        return empty_standard_gdf()

    gdf = normalise_input_columns(gdf)
    gdf = validate_geometry(gdf)
    gdf = parse_dates(gdf)
    gdf = drop_missing_critical_data(gdf, label)
    gdf = clean_standard_fields(gdf)
    gdf = add_missing_standard_columns(gdf)
    gdf = gdf[STANDARD_ORDER]
    gdf = gpd.GeoDataFrame(gdf, geometry=GEOM_COL, crs=getattr(gdf, "crs", "EPSG:4326"))
    return clean_invalid_rows(gdf, label=label)


# Fix known observer naming issue and drop rows before the minimum calendar year.
def clean_invalid_rows(gdf, min_year=2020, label="gdf"):
    if "obs" in gdf.columns:
        gdf["obs"] = clean_text_series(gdf["obs"], title_case=True)
        gdf["obs"] = gdf["obs"].str.replace("Banner Robinson", "Jackson Robinson", case=False, regex=True)

    before = len(gdf)
    gdf["year1"] = pd.to_numeric(gdf["year1"], errors="coerce")
    gdf = gdf[gdf["year1"] >= min_year]
    print(f"Dropped {before - len(gdf)} records from {label} with calendar year before {min_year}")
    return gdf


# Ensures geometry is valid point geometry in EPSG:4326 for mapping in the app.
def validate_geometry(gdf):
    if GEOM_COL not in gdf.columns:
        print("Warning: No geometry column found.")
        return gdf

    gdf[GEOM_COL] = gdf[GEOM_COL].apply(parse_geometry)
    gdf[GEOM_COL] = gdf[GEOM_COL].apply(lambda g: g if isinstance(g, Point) and not g.is_empty else None)
    gdf = gpd.GeoDataFrame(gdf, geometry=GEOM_COL)

    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")
    return gdf


# Parses dates and derives calendar year, month, day, and ecological sampling year.
def parse_dates(gdf):
    if all(col in gdf.columns for col in ["year1", "month", "day"]):
        final_date = date_from_split_columns(gdf)
        if DATE_COL in gdf.columns:
            final_date = final_date.combine_first(standardise_date_series(gdf[DATE_COL]))
        return apply_date_columns(gdf, final_date)

    if DATE_COL in gdf.columns:
        return apply_date_columns(gdf, standardise_date_series(gdf[DATE_COL]))
    return gdf


# Builds dates from split year/month/day columns to avoid day/month parsing ambiguity.
def date_from_split_columns(gdf):
    return pd.to_datetime(
        {
            "year": pd.to_numeric(gdf["year1"], errors="coerce"),
            "month": pd.to_numeric(gdf["month"], errors="coerce"),
            "day": pd.to_numeric(gdf["day"], errors="coerce"),
        },
        errors="coerce",
    )


# Writes the final date values back into Date, year1, month, day, and ecological year fields.
def apply_date_columns(gdf, dates):
    gdf[DATE_COL] = dates
    gdf["year1"] = dates.dt.year
    gdf["month"] = dates.dt.month
    gdf["day"] = dates.dt.day
    gdf["year"] = gdf.apply(calculate_sampling_year, axis=1)
    return gdf


# Calculate ecological sampling year in YYYY-YY format based on May 1 - April 30.
def calculate_sampling_year(row):
    if pd.isna(row[DATE_COL]):
        return None
    year, month = row[DATE_COL].year, row[DATE_COL].month
    if month >= 5:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


# Full pipeline: backup the main GPKG, merge student GPKGs, update species info, remove duplicates, and save.
def run_pipeline(student_gpkgs_directory, species_csv, main_file, backup_folder):
    try:
        create_main_copy(main_file, backup_folder)
        main_gdf = read_gpkg(main_file)
        student_gdfs = read_student_gpkgs(student_gpkgs_directory)
        combined = standardise(merge_collected_data(main_gdf, student_gdfs))
        combined = update_species_info(combined, species_csv, "combined gdf")
        combined = detect_and_remove_duplicates(combined, "combined gdf")
        save_main_gpkg(combined, main_file)
    except Exception as e:
        print(f"Error during pipeline execution: {e}")


# Fuzzy match species against the species CSV, then map English name and Taxa from the master list.
def match_species_in_csv(gdf, species_csv):
    species_df = pd.read_csv(species_csv, encoding="ISO-8859-1")
    species_df["_norm"] = species_df["species"].apply(normalise_species_name)
    species_df = species_df[species_df["_norm"] != ""]

    dupes = species_df["_norm"][species_df["_norm"].duplicated()].unique()
    if len(dupes) > 0:
        print(f"Warning: Duplicate species in CSV (keeping first): {', '.join(dupes)}")
        species_df = species_df.drop_duplicates(subset="_norm", keep="first")

    species_map = species_df.set_index("_norm")[["type", "english_name"]].to_dict(orient="index")
    gdf["_norm"] = gdf["species"].apply(normalise_species_name)
    gdf["Taxa"] = gdf["_norm"].map(lambda x: species_map.get(x, {}).get("type", None))
    gdf["english_name"] = gdf["_norm"].map(lambda x: species_map.get(x, {}).get("english_name", None))

    unmatched = sorted(set(gdf["_norm"]) - set(species_map.keys()))
    gdf = gdf.drop(columns=["_norm"])
    return gdf, unmatched


# Update species-derived fields and drop records whose species are not in the master species CSV.
def update_species_info(gdf, species_csv, label="gdf"):
    try:
        gdf, unmatched = match_species_in_csv(gdf, species_csv)
        if unmatched:
            print(f"Warning: Dropping {len(unmatched)} unmatched species from {label}:")
            for sp in unmatched[:10]:
                print(f"  - {sp}")
            if len(unmatched) > 10:
                print(f"  - ... and {len(unmatched) - 10} more")
            gdf = gdf[~gdf["species"].apply(normalise_species_name).isin(unmatched)]

        missing_taxa = gdf["Taxa"].isna().sum()
        if missing_taxa:
            print(f"Warning: {missing_taxa} records from {label} have missing Taxa after species matching")
    except Exception as e:
        print(f"Error updating species info: {e}")
    return gdf


# Makes a backup copy of the main GPKG before merging new data.
def create_main_copy(main_file, backup_folder):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name, ext = os.path.splitext(os.path.basename(main_file))
    destination = os.path.join(backup_folder, f"{name}_({timestamp}){ext}")
    with open(main_file, "rb") as src, open(destination, "wb") as dst:
        dst.write(src.read())
    print(f"Backup created: {destination}")


# Pick the layer that matches the file name when present; otherwise use the first layer.
def choose_gpkg_layer(gpkg_file):
    layers = fiona.listlayers(gpkg_file)
    preferred_layer = os.path.splitext(os.path.basename(gpkg_file))[0]
    return preferred_layer if preferred_layer in layers else layers[0]


# Read one GPKG file into the standard app format.
def read_gpkg(gpkg_file):
    layer = choose_gpkg_layer(gpkg_file)
    print(f"Reading GPKG: {os.path.basename(gpkg_file)} | layer: {layer}")
    gdf = gpd.read_file(gpkg_file, layer=layer)
    return standardise(gdf, label=os.path.basename(gpkg_file))


# Read all student GPKG files from a directory and combine them into one GeoDataFrame.
def read_student_gpkgs(directory):
    combined_data = []
    for path in glob.glob(os.path.join(directory, "*.gpkg")):
        try:
            gdf = read_gpkg(path)
            combined_data.append(gdf)
            print(f"Loaded {len(gdf)} records from {os.path.basename(path)}")
        except Exception as e:
            print(f"Skipping {os.path.basename(path)}: {e}")

    if not combined_data:
        print("No valid student data found.")
        return empty_standard_gdf()

    return gpd.GeoDataFrame(pd.concat(combined_data, ignore_index=True), geometry=GEOM_COL, crs="EPSG:4326")


# Merge all student GeoDataFrames into the main GPKG GeoDataFrame.
def merge_collected_data(main_gdf, student_gdfs):
    if student_gdfs is None or student_gdfs.empty:
        print("No student data to merge.")
        return main_gdf
    print(f"Main data: {len(main_gdf)} records")
    print(f"Student data: {len(student_gdfs)} records")
    combined = pd.concat([main_gdf, student_gdfs], ignore_index=True)
    print(f"Combined before duplication removal: {len(combined)} records")
    return combined


# Detect and remove duplicate records from the combined GeoDataFrame.
def detect_and_remove_duplicates(gdf, label="gdf"):
    if gdf is None or gdf.empty:
        return empty_standard_gdf()

    group_cols = ["year1", "month", "day", "geom", "species"]
    existing_cols = [col for col in group_cols if col in gdf.columns]
    before = len(gdf)

    if "geom" in gdf.columns:
        gdf["geom"] = gdf["geom"].apply(
            lambda g: Point(round(g.x, 10), round(g.y, 10))
            if isinstance(g, BaseGeometry) and g.geom_type == "Point"
            else g
        )

    gdf = gdf.drop_duplicates(subset=existing_cols, keep="first")
    print(f"Removed {before - len(gdf)} duplicate records from {label}")
    return gdf


# Save the updated main GeoDataFrame back to a GPKG file.
def save_main_gpkg(gdf, main_file):
    layer_name = os.path.splitext(os.path.basename(main_file))[0]
    folder = os.path.dirname(main_file)
    temp_file = os.path.join(folder, f"{layer_name}_tmp_write.gpkg")
    if os.path.exists(temp_file):
        os.remove(temp_file)
    gdf.to_file(temp_file, layer=layer_name, driver="GPKG")
    os.replace(temp_file, main_file)
    print(f"Main GPKG updated: {len(gdf)} total records")
    print(f"Saved to: {main_file}")





