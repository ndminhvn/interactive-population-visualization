import shapefile
import vtk

# Input Shapefile path
shp_path = "./data/VA_counties/VA_Counties.shp"
output_path = "./data/virginia_counties_fixed.vtp"

# Read shapefile
sf = shapefile.Reader(shp_path)
shapes = sf.shapes()
records = sf.records()

fields = [f[0] for f in sf.fields[1:]]

# Find name field
possible_fields = ["NAMELSAD", "NAME", "COUNTY", "COUNTYNAME"]
name_idx = None
for field in possible_fields:
    if field in fields:
        name_idx = fields.index(field)
        chosen_field = field
        break

if name_idx is None:
    raise Exception("No suitable name field found in shapefile")

print("Using field:", chosen_field)

# VTK polydata components
points = vtk.vtkPoints()
polys = vtk.vtkCellArray()
name_array = vtk.vtkStringArray()
name_array.SetName("CountyName")

for rec, shape in zip(records, shapes):

    name = rec[name_idx]
    name_array.InsertNextValue(name)

    start = points.GetNumberOfPoints()

    for pt in shape.points:
        points.InsertNextPoint(pt[0], pt[1], 0)

    poly = vtk.vtkPolygon()
    poly.GetPointIds().SetNumberOfIds(len(shape.points))

    for i in range(len(shape.points)):
        poly.GetPointIds().SetId(i, start + i)

    polys.InsertNextCell(poly)

# Build polydata
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetPolys(polys)
polydata.GetCellData().AddArray(name_array)

# Write VTP (XML format)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_path)
writer.SetInputData(polydata)
writer.Write()

print("✔️ Successfully written:", output_path)
