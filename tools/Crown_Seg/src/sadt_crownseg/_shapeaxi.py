"""shapeaxi's own code, extracted verbatim.

Everything below is copied unchanged from shapeaxi 2.0.3 by AST, not retyped,
so this file can be diffed mechanically against a future release. shapeaxi is
by the DCBIA-OrthoLab group (https://github.com/DCBIA-OrthoLab/ShapeAXI) and
carries their licence; this is a vendored subset, not a fork.

It is vendored rather than imported because `pip install shapeaxi` requires
pytorch3d, which has no Windows wheel, and because shapeaxi's import chain
pulls in lightning, ocnn, h5py, scikit-learn, torchvision, pillow, nibabel,
fslpy and itk for two small classes this pipeline uses. The functions here are
the ones the dental segmentation actually executes.

One edit, marked in place: ReadSurf's .off and .gii branches are removed, as
they need OFFReader, nibabel and fslpy and no intraoral scan uses them.

This copy comes from the lab's `iosseg` package. Every function other than
ReadSurf was compared, by AST dump, with the same function in shapeaxi 2.0.2
(utils.py and post_process.py) and is identical; ReadSurf differs by the edit
above only. `engine.py` is the only module that calls into it.
"""
import math
import os

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# =========================== shapeaxi/utils.py =====================================

def ReadSurf(fileName):
    # EDIT: the .off and .gii branches are removed; they need
    # OFFReader, nibabel and fslpy, and no intraoral scan uses them.

    fname, extension = os.path.splitext(fileName)    
    extension = extension.lower()    
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
                obj_import.SetTexturePath(textures_path)
                    

            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()
    else:
        raise Exception("File format not supported")
    
    return surf

def Write(vtkdata, output_name, print_out = True):
    outfilename = output_name
    if print_out == True:
        print("Writing:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()

def normalize_points(poly, radius=1.0):
    polypoints = poly.GetPoints()
    for pid in range(polypoints.GetNumberOfPoints()):
        spoint = polypoints.GetPoint(pid)
        spoint = np.array(spoint)
        norm = np.linalg.norm(spoint)
        spoint = spoint/norm * radius
        polypoints.SetPoint(pid, spoint)
    poly.SetPoints(polypoints)
    return poly

class LinearSubdivisionFilter:
    InputData = None
    Output = None
    NumberOfSubdivisions = 1

    def SetInputData(self, polydata):
        self.InputData = polydata

    def GetOutput(self):
        return self.Output

    def SetNumberOfSubdivisions (self, subdivisions):
        self.NumberOfSubdivisions = subdivisions

    def Update(self):
        self.GenerateData()

    def GenerateData(self):

        if self.InputData:

            inputpolydata = self.InputData
            subdivisionlevel = self.NumberOfSubdivisions
            inputpolydata_points = inputpolydata.GetPoints()

            subdiv_poly = vtk.vtkPolyData()
            subdiv_points = vtk.vtkPoints()
            subdiv_cellarray = vtk.vtkCellArray()

            incremental = vtk.vtkIncrementalOctreePointLocator()
            incremental.SetDataSet(inputpolydata)
            incremental.BuildLocator()
            incremental.InitPointInsertion(subdiv_points, inputpolydata.GetBounds())

            # Iterate over the cells in the polydata
            # The idea is to linearly divide every cell according to the subdivision level
            for cellid in range(inputpolydata.GetNumberOfCells()):
                idlist = vtk.vtkIdList()
                inputpolydata.GetCellPoints(cellid, idlist)
                
                # For every cell we create a new poly data, i.e, bigger triangle with the interpolated triangles inside
                if(idlist.GetNumberOfIds() != 3):
                    raise Exception("Only triangle meshes are supported. Convert your mesh to triangles!", idlist.GetNumberOfIds())

                # Get the triangle points from the current cell
                p1 = np.array(inputpolydata_points.GetPoint(idlist.GetId(0)))
                p2 = np.array(inputpolydata_points.GetPoint(idlist.GetId(1)))
                p3 = np.array(inputpolydata_points.GetPoint(idlist.GetId(2)))

                # Calculate the derivatives according to the level
                dp12 = (p2 - p1)/subdivisionlevel
                dp13 = (p3 - p1)/subdivisionlevel
                
                # Interpolate the points
                idlist_subdiv = []
                for s13 in range(0, subdivisionlevel + 1):
                    for s12 in range(0, subdivisionlevel + 1 - s13):
                        interp = p1 + s12*dp12 + s13*dp13
                        outid = incremental.IsInsertedPoint(interp)
                        if outid == -1:
                            outid = incremental.InsertNextPoint(interp)
                        idlist_subdiv.append(outid)

                # Using the interpolated points, create the cells, i.e., triangles
                id1 = -1
                for s13 in range(0, subdivisionlevel):
                    id1 += 1
                    for s12 in range(0, subdivisionlevel - s13):
                        
                        id2 = id1 + 1
                        id3 = id1 + subdivisionlevel + 1 - s13
                        id4 = id3 + 1

                        triangle = vtk.vtkTriangle()
                        triangle.GetPointIds().SetId(0, idlist_subdiv[id1]);
                        triangle.GetPointIds().SetId(1, idlist_subdiv[id2]);
                        triangle.GetPointIds().SetId(2, idlist_subdiv[id3]);

                        subdiv_cellarray.InsertNextCell(triangle)

                        if s12 < subdivisionlevel - s13 - 1:
                            triangle = vtk.vtkTriangle()
                            triangle.GetPointIds().SetId(0, idlist_subdiv[id2]);
                            triangle.GetPointIds().SetId(1, idlist_subdiv[id4]);
                            triangle.GetPointIds().SetId(2, idlist_subdiv[id3]);
                            subdiv_cellarray.InsertNextCell(triangle)

                        id1 += 1

            # Remove duplicate points (if you were paying attention, you know there are a lot of repetitions in every triangle edge)
            subdiv_poly.SetPoints(subdiv_points)
            subdiv_poly.SetPolys(subdiv_cellarray)

            # Return the subdivied polydata
            self.Output = subdiv_poly

def CreateIcosahedronSubdivided(radius, sl):
    icosahedronsource = vtk.vtkPlatonicSolidSource()
    icosahedronsource.SetSolidTypeToIcosahedron()
    icosahedronsource.Update()
    icosahedron = icosahedronsource.GetOutput()
    icosahedron = normalize_points(icosahedron, radius)

    subdivfilter = LinearSubdivisionFilter()
    subdivfilter.SetInputData(icosahedron)
    subdivfilter.SetNumberOfSubdivisions(sl)
    subdivfilter.Update()
    subdivfilter = subdivfilter.GetOutput()
    subdivfilter = normalize_points(subdivfilter, radius)

    return subdivfilter

def PolyDataToNumpy_v_f(surf):

    vtk.vtkObject.GlobalWarningDisplayOff()
    verts = vtk_to_numpy(surf.GetPoints().GetData())
    faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
    
    return verts, faces

def PolyDataToTensors_v_f(surf, device='cpu'):

    verts, faces, = PolyDataToNumpy_v_f(surf)
    
    verts = torch.tensor(verts).to(torch.float32).to(device)
    faces = torch.tensor(faces).to(torch.int64).to(device)
    
    return verts, faces

def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOn();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()

def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def ScaleSurf(surf, mean_arr = None, scale_factor = None, copy=True):
    if(copy):
        surf_copy = vtk.vtkPolyData()
        surf_copy.DeepCopy(surf)
        surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = vtk_to_numpy(shapedatapoints.GetData())
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1.0/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    shapedatapoints.SetData(numpy_to_vtk(shape_points))

    return surf, mean_arr, scale_factor

def GetUnitSurf(surf, mean_arr = None, scale_factor = None, copy=True):
    unit_surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor, copy)
    return unit_surf

def GetNeighbors(vtkdata, pid):
    cells_id = vtk.vtkIdList()
    vtkdata.GetPointCells(pid, cells_id)
    neighbor_pids = []

    for ci in range(cells_id.GetNumberOfIds()):
        points_id_inner = vtk.vtkIdList()
        vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
        for pi in range(points_id_inner.GetNumberOfIds()):
            pid_inner = points_id_inner.GetId(pi)
            if pid_inner != pid:
                neighbor_pids.append(pid_inner)

    return np.unique(neighbor_pids).tolist()

def GetNeighborIds(vtkdata, pid, labels, label, pid_visited):
    cells_id = vtk.vtkIdList()
    vtkdata.GetPointCells(pid, cells_id)
    neighbor_pids = []

    for ci in range(cells_id.GetNumberOfIds()):
        points_id_inner = vtk.vtkIdList()
        vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
        for pi in range(points_id_inner.GetNumberOfIds()):
            pid_inner = points_id_inner.GetId(pi)
            if labels.GetTuple(pid_inner)[0] == label and pid_inner != pid and pid_visited[pid_inner] == 0:
                pid_visited[pid_inner] = 1
                neighbor_pids.append(pid_inner)

    return np.unique(neighbor_pids).tolist()


# =========================== shapeaxi/post_process.py ==============================

def ConnectedRegion(vtkdata, pid, labels, label, pid_visited):

	neighbor_pids = GetNeighborIds(vtkdata, pid, labels, label, pid_visited)
	all_connected_pids = [pid]
	all_connected_pids.extend(neighbor_pids)

	while len(neighbor_pids):
		npid = neighbor_pids.pop()
		next_neighbor_pids = GetNeighborIds(vtkdata, npid, labels, label, pid_visited)
		neighbor_pids.extend(next_neighbor_pids)
		all_connected_pids = np.append(all_connected_pids, next_neighbor_pids)

	return np.unique(all_connected_pids)

def NeighborLabel(vtkdata, labels, label, connected_pids):
	neighbor_ids = []
	
	for pid in connected_pids:
		cells_id = vtk.vtkIdList()
		vtkdata.GetPointCells(int(pid), cells_id)
		for ci in range(cells_id.GetNumberOfIds()):
			points_id_inner = vtk.vtkIdList()
			vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
			for pi in range(points_id_inner.GetNumberOfIds()):
				pid_inner = points_id_inner.GetId(pi)
				if labels.GetTuple(pid_inner)[0] != label:
					neighbor_ids.append(pid_inner)

	neighbor_ids = np.unique(neighbor_ids)
	neighbor_labels = []

	for nid in neighbor_ids:
		neighbor_labels.append(labels.GetTuple(nid)[0])
	
	if len(neighbor_labels) > 0:
		return max(neighbor_labels, key=neighbor_labels.count)
	return -1

def RemoveIslands(vtkdata, labels, label, min_count,ignore_neg1 = False):

	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			if connected_pids.shape[0] < min_count:
				neighbor_label = NeighborLabel(vtkdata, labels, label, connected_pids)
				if ignore_neg1 == True and neighbor_label != -1:
					for cpid in connected_pids:
						labels.SetTuple(int(cpid), (neighbor_label,))

def ErodeLabel(vtkdata, labels, label, ignore_label=None,iterations=math.inf, target=None ):
	
	pid_labels = []
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			pid_labels.append(pid)

	while pid_labels and iterations !=0:
		pid_labels_remain = pid_labels
		pid_labels = []

		all_neighbor_pids = []
		all_neighbor_labels = []

		while pid_labels_remain:

			pid = pid_labels_remain.pop()

			neighbor_pids = GetNeighbors(vtkdata, pid)
			is_neighbor = False

			for npid in neighbor_pids:
				neighbor_label = labels.GetTuple(npid)[0]
				if neighbor_label != label and (ignore_label == None or neighbor_label != ignore_label) and (target == None or neighbor_label == target):

					all_neighbor_pids.append(pid)
					all_neighbor_labels.append(neighbor_label)
					is_neighbor = True
					break

			if not is_neighbor:
				pid_labels.append(pid)

		if(all_neighbor_pids):
			for npid, nlabel in zip(all_neighbor_pids, all_neighbor_labels):
				labels.SetTuple(int(npid), (nlabel,))
		else:
			break
		iterations -= 1

def DilateLabel(vtkdata, labels, label, iterations=2, dilateOverTarget=False, target=None):
	
	pid_labels = []

	while iterations > 0:
		#Get all neighbors to the 'label' that have a different label
		all_neighbor_labels = []
		for pid in range(labels.GetNumberOfTuples()):
			if labels.GetTuple(pid)[0] == label:
				neighbor_pids = GetNeighbors(vtkdata, pid)
				pid_labels.append(pid)

				for npid in neighbor_pids:
					neighbor_label = labels.GetTuple(npid)[0]

					if dilateOverTarget:
						if neighbor_label == target:
							all_neighbor_labels.append(npid)
					else:
						if neighbor_label != label:
							all_neighbor_labels.append(npid)

		#Dilate them, i.e., change the value to label
		for npid in all_neighbor_labels:
			labels.SetTuple(int(npid), (label,))

		iterations -= 1

def Threshold(vtkdata, labels, threshold_min, threshold_max, invert=False):
	
	threshold = vtk.vtkThreshold()
	threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labels)
	threshold.SetInputData(vtkdata)
	# threshold.ThresholdBetween(threshold_min,threshold_max)
	threshold.SetLowerThreshold(threshold_min)
	threshold.SetUpperThreshold(threshold_max)
	threshold.SetInvert(invert)
	threshold.Update()

	geometry = vtk.vtkGeometryFilter()
	geometry.SetInputData(threshold.GetOutput())
	geometry.Update()
	return geometry.GetOutput()

