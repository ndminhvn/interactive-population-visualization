import vtk

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("./data/virginia_counties_fixed.vtp")
reader.Update()

poly = reader.GetOutput()
cell_data = poly.GetCellData()

print("Num polygons:", poly.GetNumberOfCells())
print("Num cell arrays:", cell_data.GetNumberOfArrays())

for i in range(cell_data.GetNumberOfArrays()):
    arr = cell_data.GetAbstractArray(i)
    if arr:
        print(f"Array {i} name:", arr.GetName())
        print(f"  Number of values:", arr.GetNumberOfValues())
    else:
        print(f"Array {i}: None")
