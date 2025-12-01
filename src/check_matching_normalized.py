import pandas as pd
import vtk
from normalize_name import normalize_name  # or paste directly

# -------------------- Load CSV --------------------
df = pd.read_csv("./data/virginia_population.csv")
csv_raw = df["County"].tolist()
csv_norm = [normalize_name(x) for x in csv_raw]

# -------------------- Load VTP --------------------
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("./data/virginia_counties_fixed.vtp")
reader.Update()

poly = reader.GetOutput()
cell_data = poly.GetCellData()
arr = cell_data.GetAbstractArray(0)

vtp_raw = [arr.GetValue(i) for i in range(arr.GetNumberOfValues())]
vtp_norm = [normalize_name(x) for x in vtp_raw]

# -------------------- Compare --------------------
csv_set = set(csv_norm)
vtp_set = set(vtp_norm)

missing_in_vtk = csv_set - vtp_set
missing_in_csv = vtp_set - csv_set

print("=== CSV names NOT in VTK ===")
print(missing_in_vtk)

print("\n=== VTK names NOT in CSV ===")
print(missing_in_csv)
