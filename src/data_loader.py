import os
import pandas as pd
import vtk
from typing import Dict, List, Any, Tuple
from normalize_name import normalize_name

# ----------------------------------------------------------------------
# Paths (relative to this file)
# ----------------------------------------------------------------------
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

CSV_PATH = os.path.join(DATA_DIR, "virginia_population.csv")
VTP_PATH = os.path.join(DATA_DIR, "virginia_counties_fixed.vtp")


# ----------------------------------------------------------------------
# Population data loading
# ----------------------------------------------------------------------
def load_population_data(csv_path: str = CSV_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the Virginia population CSV and return:
      - DataFrame with a normalized county column
      - List of year columns as strings ("2010" ... "2019")
    """
    df = pd.read_csv(csv_path)

    if "County" not in df.columns:
        raise ValueError("Expected a 'County' column in the CSV.")

    # Normalize county names for matching with VTK
    df["County_norm"] = df["County"].apply(normalize_name)

    # Population years (2010–2019)
    year_cols = [str(y) for y in range(2010, 2020)]
    for y in year_cols:
        if y not in df.columns:
            raise ValueError(f"Expected year column '{y}' in CSV.")

    return df, year_cols


# ----------------------------------------------------------------------
# VTK PolyData loading
# ----------------------------------------------------------------------
def load_virginia_polydata(vtp_path: str = VTP_PATH) -> vtk.vtkPolyData:
    """
    Load the Virginia counties VTP file and return the vtkPolyData.
    Assumes a string cell-data array called 'CountyName'.
    """
    if not os.path.exists(vtp_path):
        raise FileNotFoundError(f"VTP file not found at: {vtp_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()

    poly = reader.GetOutput()
    if poly is None:
        raise RuntimeError("Failed to read VTP file.")

    cell_data = poly.GetCellData()
    if cell_data.GetNumberOfArrays() == 0:
        raise RuntimeError("VTP has no cell data arrays.")

    # Look for our CountyName array
    name_array = None
    for i in range(cell_data.GetNumberOfArrays()):
        arr = cell_data.GetAbstractArray(i)
        if arr and arr.GetName() == "CountyName":
            name_array = arr
            break

    if name_array is None:
        raise RuntimeError("Cell data array 'CountyName' not found in VTP.")

    return poly


# ----------------------------------------------------------------------
# County index: link CSV ↔ VTK polygons
# ----------------------------------------------------------------------
def build_county_index(
    poly: vtk.vtkPolyData, df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from normalized county name to:
      {
        'csv_row': int,                # row index in DataFrame
        'csv_name': str,               # original CSV 'County' name
        'vtk_cell_ids': List[int],     # one or more polygon cell IDs
      }

    This handles cases where a single county has multiple polygons.
    """
    cell_data = poly.GetCellData()
    name_array = None

    for i in range(cell_data.GetNumberOfArrays()):
        arr = cell_data.GetAbstractArray(i)
        if arr and arr.GetName() == "CountyName":
            name_array = arr
            break

    if name_array is None:
        raise RuntimeError("CountyName array not found in polydata cell data.")

    n_cells = poly.GetNumberOfCells()

    # --- Build mapping: normalized_name -> list of VTK cell IDs ---
    vtk_name_to_cells: Dict[str, List[int]] = {}
    for cell_id in range(n_cells):
        raw_name = name_array.GetValue(cell_id)
        norm = normalize_name(raw_name)
        vtk_name_to_cells.setdefault(norm, []).append(cell_id)

    # --- Build mapping: normalized_name -> CSV row index + display name ---
    csv_name_to_row: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in df.iterrows():
        norm = row["County_norm"]
        csv_name_to_row[norm] = {
            "csv_row": row_idx,
            "csv_name": row["County"],
        }

    # --- Merge both sources into a unified index ---
    index: Dict[str, Dict[str, Any]] = {}

    all_keys = set(vtk_name_to_cells.keys()) | set(csv_name_to_row.keys())
    for key in all_keys:
        vtk_cells = vtk_name_to_cells.get(key)
        csv_info = csv_name_to_row.get(key)

        if vtk_cells is None or csv_info is None:
            continue

        index[key] = {
            "csv_row": csv_info["csv_row"],
            "csv_name": csv_info["csv_name"],
            "vtk_cell_ids": vtk_cells,
        }

    return index


# ----------------------------------------------------------------------
# Convenience helper for the rest of the app
# ----------------------------------------------------------------------
def load_all_data() -> Dict[str, Any]:
    """
    Load everything needed by the Trame app in one call.

    Returns a dict:
      {
        'df': DataFrame,
        'year_cols': [year strings],
        'poly': vtkPolyData,
        'county_index': {normalized_name: {csv_row, csv_name, vtk_cell_ids}},
        'county_list': [display names in sorted order],
      }
    """
    df, year_cols = load_population_data()
    poly = load_virginia_polydata()
    county_index = build_county_index(poly, df)

    # Create a sorted list of display names for dropdown
    # (We use csv_name so it shows "Accomack County", "Hampton city", etc.)
    county_list = sorted({info["csv_name"] for info in county_index.values()})

    return {
        "df": df,
        "year_cols": year_cols,
        "poly": poly,
        "county_index": county_index,
        "county_list": county_list,
    }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = load_all_data()
    print("Loaded DataFrame shape:", data["df"].shape)
    print("Year columns:", data["year_cols"])
    print("Number of VTK cells:", data["poly"].GetNumberOfCells())
    print("Number of counties in index:", len(data["county_index"]))
    print("First 10 counties:", data["county_list"][:10])
