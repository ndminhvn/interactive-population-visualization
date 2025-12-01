import vtk
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# Create base color array (light gray for all counties)
# ----------------------------------------------------------------------
def create_base_colors(
    poly: vtk.vtkPolyData,
    base_rgb: Tuple[int, int, int] = (235, 235, 235),
) -> vtk.vtkUnsignedCharArray:
    """
    Create a vtkUnsignedCharArray with one RGB color per polygon (cell).
    All counties start with the same light-gray fill color.
    """
    n_cells = poly.GetNumberOfCells()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    r, g, b = base_rgb
    for _ in range(n_cells):
        colors.InsertNextTuple3(r, g, b)

    # Attach as cell scalars so mapper can use them directly
    poly.GetCellData().SetScalars(colors)

    return colors


# ----------------------------------------------------------------------
# Build the actor + mapper for the Virginia map
# ----------------------------------------------------------------------
def create_virginia_actor(
    poly: vtk.vtkPolyData,
    base_rgb: Tuple[int, int, int] = (235, 235, 235),
) -> Tuple[vtk.vtkActor, vtk.vtkPolyDataMapper, vtk.vtkUnsignedCharArray]:
    """
    Given the Virginia counties vtkPolyData, create:
      - a mapper using cell colors
      - an actor with nice edge styling
      - and the underlying color array (for highlighting)
    """
    colors = create_base_colors(poly, base_rgb=base_rgb)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Polygon fill is driven by the colors array,
    # but we can still control edge styling via the actor property
    prop = actor.GetProperty()
    prop.SetEdgeVisibility(True)
    prop.SetEdgeColor(0, 0, 0)  # dark grey borders
    prop.SetLineWidth(3.0)
    prop.SetOpacity(1.0)

    return actor, mapper, colors


# ----------------------------------------------------------------------
# Create a renderer that shows the Virginia map
# ----------------------------------------------------------------------
def create_virginia_renderer(actor: vtk.vtkActor) -> vtk.vtkRenderer:
    """
    Build a simple 2D-style renderer for the flat Virginia map.
    """
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)  # white background

    # Make it feel more like a flat 2D map
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    renderer.ResetCamera()
    return renderer


# ----------------------------------------------------------------------
# Highlighting logic
# ----------------------------------------------------------------------
def highlight_county(
    poly: vtk.vtkPolyData,
    colors: vtk.vtkUnsignedCharArray,
    cell_ids: Optional[List[int]],
    base_rgb: Tuple[int, int, int] = (235, 235, 235),
    highlight_rgb: Tuple[int, int, int] = (255, 215, 0),
):
    """
    Highlight the given list of VTK cell IDs (for a selected county).

    - Resets all counties to base_rgb (light gray)
    - Applies highlight_rgb (e.g., yellow) to the selected county's polygons

    cell_ids can include multiple polygons if a county is disjoint.
    """
    n_cells = poly.GetNumberOfCells()
    r0, g0, b0 = base_rgb
    r1, g1, b1 = highlight_rgb

    # 1) Reset all cells to base color
    for cid in range(n_cells):
        colors.SetTuple3(cid, r0, g0, b0)

    # 2) Highlight selected county polygons
    if cell_ids is not None:
        for cid in cell_ids:
            if 0 <= cid < n_cells:
                colors.SetTuple3(cid, r1, g1, b1)

    # Tell VTK that the colors have changed
    colors.Modified()


# ----------------------------------------------------------------------
# Optional convenience: reset all highlights
# ----------------------------------------------------------------------
def reset_highlight(
    poly: vtk.vtkPolyData,
    colors: vtk.vtkUnsignedCharArray,
    base_rgb: Tuple[int, int, int] = (235, 235, 235),
):
    """
    Reset the entire map back to the base color (no highlighted county).
    """
    n_cells = poly.GetNumberOfCells()
    r0, g0, b0 = base_rgb

    for cid in range(n_cells):
        colors.SetTuple3(cid, r0, g0, b0)

    colors.Modified()


# ----------------------------------------------------------------------
# Simple standalone test (optional)
# Run:  python src/map_utils.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # This is just a quick VTK-only test (not using trame).
    import os
    from data_loader import load_virginia_polydata

    HERE = os.path.dirname(__file__)
    poly = load_virginia_polydata()

    actor, mapper, colors = create_virginia_actor(poly)
    renderer = create_virginia_renderer(actor)

    # Just highlight the first polygon to visually test
    highlight_county(poly, colors, [0])

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(800, 600)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    ren_win.Render()
    iren.Start()
