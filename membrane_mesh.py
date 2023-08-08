import numpy as np
import matplotlib.pyplot as plt
import voltools as vt
from voltools.utils import rotation_matrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import distance
from skimage import measure

# replace the firt part!
epsilon = 0.0000005

def arctan(sinX, cosX):
    """
    compute arctan as function of sin(x) and cos(x)
    @param sinX: sine
    @type sinX: float
    @param cosX: cosine
    @type cosX: float
    @return: atan - automatically map to correct segment
    @rtype: float
    """
    from math import atan, pi
    if cosX == 0:
        if sinX >= 0:
           x = pi/2.
        else:
           x = 3.*pi/2.
    else:
        x = atan(sinX/cosX)
    if cosX < 0:
        x = x + pi
    if x < 0:
        x = x + 2.*pi
    return x


def matToZXZ(rotMatrix):
    """
    matToZXZ : Converts a rotation matrix to zxz angles z1,z2,x.
    @param rotMatrix: The rotation matrix
    @param inRad: Do you want the returned angles be in radians?
    @return: [z1,z2,x] 
    @author: Friedrich Forster
    """
    import math
    from numpy import sign

    # determine X-rotation angle
    cosX = rotMatrix[2, 2]
    if cosX >= 1.:
        x = 0.
    elif cosX <= -1.:
        x = math.pi
    else:
        x = math.acos(rotMatrix[2, 2])
    sinX = math.sin(x)
    if abs(sinX) >= epsilon:
        z1 = arctan(sign(sinX) * rotMatrix[2, 0], sign(sinX) * rotMatrix[2, 1])
        z2 = arctan(sign(sinX) * rotMatrix[0, 2], -sign(sinX) * rotMatrix[1, 2])
    else:
        # set z1=0
        z1 = 0.
        cosZ2 = rotMatrix[0, 0]
        if cosZ2 > 1.:
            z2 = 0
        elif cosZ2 < -1.:
            z2 = math.pi
        else:
            z2 = math.acos(cosZ2)
        # x=0 deg
        if cosX > 0:
            if rotMatrix[0, 1] > 0.:
                z2 = 2 * math.pi - z2
        # x=180 deg
        else:
            if rotMatrix[0, 1] > 0.:
                z2 = 2 * math.pi - z2
                
    return np.rad2deg(z1), np.rad2deg(x), np.rad2deg(z2)


# ========================================= VECTOR CLASS AND Z ROT MATRIX ==============================================
class Vector:
    # Class can be used as both a 3d coordinate, and a vector
    # TODO SCIPY probably also has a vector class
    def __init__(self, coordinates, normalize=False):
        """
        Init vector with (x,y,z) coordinates, assumes (0,0,0) origin.
        """
        assert len(coordinates) == 3, 'Invalid axis list for a 3d vector, input does not contain 3 coordinates.'
        self._axis = np.array(coordinates)
        self._zero_vector = np.all(self._axis==0)
        if normalize:
            self.normalize()

    def get(self):
        """
        Return vector in numpy array.
        """
        return self._axis

    def show(self):
        """
        Print the vector.
        """
        print(self._axis)

    def copy(self):
        """
        Return a copy of the vector (also class Vector).
        """
        return Vector(self.get())

    def inverse(self):
        """
        Inverse the vector (in place).
        """
        return Vector(self._axis * -1)

    def cross(self, other):
        """
        Get cross product of self and other Vector. Return as new vector.
        """
        return Vector([self._axis[1] * other._axis[2] - self._axis[2] * other._axis[1],
                       self._axis[2] * other._axis[0] - self._axis[0] * other._axis[2],
                       self._axis[0] * other._axis[1] - self._axis[1] * other._axis[0]])

    def dot(self, other):
        """
        Return the dot product of vectors v1 and v2, of form (x,y,z).
        Dot product of two vectors is zero if they are perpendicular.
        """
        return self._axis[0] * other._axis[0] + self._axis[1] * other._axis[1] + self._axis[2] * other._axis[2]

    def average(self, other):
        return Vector((self._axis + other._axis) / 2)

    def magnitude(self):
        """
        Calculate the magnitude (length) of vector p.
        """
        return np.sqrt(np.sum(self._axis ** 2))

    def normalize(self):
        """
        Normalize self by dividing by magnitude.
        """
        if not self._zero_vector:
            self._axis = self._axis / self.magnitude()

    def angle(self, other, degrees=False):
        """
        Get angle between self and other.
        """
        # returns angle in radians
        if self._zero_vector or other._zero_vector:
            angle = 0
        else:
            angle = np.arccos(self.dot(other) / (self.magnitude() * other.magnitude()))
        if degrees:
            return angle * 180 / np.pi
        else:
            return angle

    def rotate(self, rotation_matrix):
        """
        Rotate the vector in place by the rotation matrix.
        """
        return Vector(np.dot(self._axis, rotation_matrix))

    def _get_orthogonal_unit_vector(self):
        """
        Get some orthogonal unit vector, multiple solutions are possible. Private method used in get rotation.
        """
        # A vector orthogonal to (a, b, c) is (-b, a, 0), or (-c, 0, a) or (0, -c, b).
        if self._zero_vector:
            return Vector([1, 0, 0])  # zero vector is orthogonal to any vector
        else:
            if self._axis[2] != 0:
                x, y = 1, 1
                z = (- 1 / self._axis[2]) * (x * self._axis[0] + y * self._axis[1])
            elif self._axis[1] != 0:
                x, z = 1, 1
                y = (- 1 / self._axis[1]) * (x * self._axis[0] + z * self._axis[2])
            else:
                y, z = 1, 1
                x = (- 1 / self._axis[0]) * (y * self._axis[1] + z * self._axis[2])
            orth = Vector([x, y, z])
            orth.normalize()
            np.testing.assert_allclose(self.dot(orth), 0, atol=1e-7, err_msg='something went wrong in finding ' \
                                                                             'perpendicular vector')
            return orth

    def get_rotation(self, other, as_affine_matrix=False):
        """
        Get rotation to rotate other vector onto self. Take the transpose of result to rotate self onto other.
        """
        if self._zero_vector or other._zero_vector:
            return np.identity(3)

        nself, nother = self.copy(), other.copy()
        nself.normalize()
        nother.normalize()

        if nself.dot(nother) > 0.99999:  # if the vectors are parallel
            return np.identity(3)  # return identity
        elif nself.dot(nother) < -0.99999:  # if the vectors are opposite
            axis = nself._get_orthogonal_unit_vector()  # return 180 along whatever axis
            angle = np.pi  # 180 degrees rotation around perpendicular vector
        else:
            axis = nself.cross(nother)
            axis.normalize()
            angle = nself.angle(nother)

        x, y, z = axis.get()
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c

        m00 = c + x * x * t
        m11 = c + y * y * t
        m22 = c + z * z * t

        tmp1 = x * y * t
        tmp2 = z * s
        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2
        tmp1 = x * z * t
        tmp2 = y * s
        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2
        tmp1 = y * z * t
        tmp2 = x * s
        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2

        mat = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        if as_affine_matrix:  # make 4x4
            out = np.identity(4)
            out[:3, :3] = mat
            mat = out

        return mat


# ==============================================MEMBRANE MESH CLASS AND SUPPORT=========================================
def point_3d_in_triangle(point, v1, v2, v3):
    """
    Reference: W. Heidrich, Journal of Graphics, GPU, and Game Tools,Volume 10, Issue 3, 2005
    @param point: coordinate in 3D
    @type point: np.array(3)
    @param v1: vertices defined by xyz
    @type v1: np.array(3)
    @type v2: np.array(3)
    @type v3: np.array(3)
    @return: (projection of point on triangle plane; whether the projection is inside the triangle)
    @rtype: np.array(3), bool
    """
    type_list = [type(point), type(v1), type(v2), type(v3)]
    assert len(set(type_list)) == 1, "Input is not of same type."
    if all([t is list for t in type_list]):
        point, v1, v2, v3 = map(np.array, [point, v1, v2, v3])

    uv = v2 - v1
    vv = v3 - v1
    nv = np.cross(uv, vv)
    ov = point - v1

    gamma = np.dot(np.cross(uv, ov), nv) / np.dot(nv, nv)
    beta = np.dot(np.cross(ov, vv), nv) / np.dot(nv, nv)
    alpha = 1 - gamma - beta

    pp = alpha * v1 + beta * v2 + gamma * v3  # the projected point (pp)
    is_in_triangle = (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1)

    return pp, is_in_triangle


def point_3d_in_line(point, a, b):
    """
    @param point: coordinate in 3D
    @type point: np.array(3)
    @param a: first point of line segment
    @type a: np.array(3)
    @param b: second point of line segment
    @type b: np.array(3)
    @return: (projection of point on line (i.e. perpendicular); whether projection is within a, b)
    @rtype: np.array(3), bool
    """
    ap = point - a
    ab = b - a

    # projected point (pp)
    pp = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

    # only if distance of ( p -> a ) + ( p -> b) == (a -> b) the point is in between a and b
    if distance.euclidean(a, b) == \
            distance.euclidean(a, pp) + \
            distance.euclidean(b, pp):
        return pp, True

    return pp, False  # else


def faces_to_edges(faces, vert):
    """
    Convert faces to list of edges connected to vert, where each edge holds the indices to its adjacent faces.
    """
    # dict that maps vertex to the remaining vertices of the face
    index_dict = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    edges = []

    for face in faces:
        id = face.tolist().index(vert)  # each face can have two possible edges connected to the vert
        rem = index_dict[id]  # an edge is defined by two vertices
        e1 = [face[id], face[rem[0]]]  # => these are the two possible combinations
        e2 = [face[id], face[rem[1]]]

        # append only if not yet recorded
        if not any([True for e in edges if (e == e1 or e == e1[::-1])]):
            edges.append(e1)
        if not any([True for e in edges if (e == e2 or e == e2[::-1])]):
            edges.append(e2)

    return np.array(edges)


def get_edge_vector(edge, faces, verts):
    """
    Find the normal on an edge by averaging the normals of the two adjacent faces.
    """
    # get the indices of the faces next to the edge
    ids = (np.any(faces == edge[0], axis=1) == np.any(faces == edge[1], axis=1))
    faces_duo = faces[ids]
    assert len(faces_duo) == 2, "something is wrong in finding edge vector, could not find two connected faces"

    # find first normal
    v1, v2, v3 = verts[faces[0][0]], verts[faces[0][1]], verts[faces[0][2]]
    normal1 = Vector(np.cross(v2 - v1, v3 - v1))
    normal1.normalize()
    normal1.inverse()

    # find second normal
    v1, v2, v3 = verts[faces[1][0]], verts[faces[1][1]], verts[faces[1][2]]
    normal2 = Vector(np.cross(v2 - v1, v3 - v1))
    normal2.normalize()
    normal2.inverse()

    # calculate average
    average = normal1.average(normal2)
    average.normalize()
    return average


def find_std(face_normals):
    """
    Calculate the variation of a set of triangle normals.
    """
    # get the mean vector of all the normals: sum all vectors and divice by number of vectors
    mean_vec = Vector(np.array([n.get() for n in face_normals]).sum(axis=0) / len(face_normals))

    # calculate difference of each one to the mean
    diff_angles = np.array(list(map(lambda x: mean_vec.angle(x, degrees=True), face_normals)))

    return diff_angles.std()  # return their sigma


class MembraneMesh:
    def __init__(self, volume, cutoff=0.3, mesh_detail=2, ref_vector=[0, 0, 1], upside_down=False):
        # ensure volume is normalized between 0 and 1
        self.volume = (volume - volume.min()) / (volume.max() - volume.min())
        self.verts, self.faces, self.normals, self.values = \
            measure.marching_cubes(self.volume, level=cutoff, step_size=mesh_detail, allow_degenerate=False)
        if ref_vector == [0, 0, 1]:
            self.reference_unit_vector = None
        else:
            self.reference_unit_vector = Vector(ref_vector, normalize=True)
        if upside_down:
            self.z_axis_unit_vector = Vector([0, 0, -1], normalize=True)
        else:
            self.z_axis_unit_vector = Vector([0, 0, 1], normalize=True)

    def write_to_bild(self, file_name):
        mesh = self.verts[self.faces]
        with open(file_name, 'w') as stream:
            for i in range(mesh.shape[0]):
                v1, v2, v3 = mesh[i, 0], mesh[i, 1], mesh[i, 2]
                # stream.write(f'.move {v1[0]} {v1[1]} {v1[2]} \n')
                # stream.write(f'.draw {v2[0]} {v2[1]} {v1[2]} \n')
                # stream.write(f'.draw {v3[0]} {v3[1]} {v3[2]} \n')
                stream.write(f'.polygon {v1[0]} {v1[1]} {v1[2]} {v2[0]} {v2[1]} {v2[2]} {v3[0]} {v3[1]} {v3[2]}\n')

    def display(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(self.verts[self.faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        ax.set_xlabel(f"x-axis: {self.volume.shape[0]}")
        ax.set_ylabel(f"y-axis: {self.volume.shape[1]}")
        ax.set_zlabel(f"z-axis: {self.volume.shape[2]}")

        ax.set_xlim(0, self.volume.shape[0])
        ax.set_ylim(0, self.volume.shape[1])
        ax.set_zlim(0, self.volume.shape[2])

        plt.tight_layout()
        plt.show()

    def visualize_vectors(self, particle_vecs, membrane_vecs):
        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(self.verts[self.faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        pdata = np.array([list(c) + list(p) for c, p in particle_vecs])
        X, Y, Z, U, V, W = zip(*pdata)
        ax.quiver(X, Y, Z, [10 * u for u in U], [10 * u for u in V], [10 * u for u in W], color='red')

        mdata = np.array([list(c) + list(p) for c, p in membrane_vecs])
        X, Y, Z, U, V, W = zip(*mdata)
        ax.quiver(X, Y, Z, [10 * u for u in U], [10 * u for u in V], [10 * u for u in W], color='blue')

        ax.set_xlabel(f"x-axis: {self.volume.shape[0]}")
        ax.set_ylabel(f"y-axis: {self.volume.shape[1]}")
        ax.set_zlabel(f"z-axis: {self.volume.shape[2]}")

        ax.set_xlim(0, self.volume.shape[0])
        ax.set_ylim(0, self.volume.shape[1])
        ax.set_zlim(0, self.volume.shape[2])

        plt.tight_layout()
        plt.show()

    def find_membrane_surface(self, coordinate):
        """
        Find the membrane surface closest to a coordinate. Returns closest point on the membrane, the normal of
        the membrane at that position, and the variation of the normals around that position.
        @type coordinate: np.array(3)
        @return: membrane point, membrane normal, standard deviation
        @rtype: np.array(3), pytom.simulation.membrane.Vector, float
        """
        # distance to each vertex in the triangle mesh
        distance_vector = np.sqrt(np.sum(np.subtract(self.verts, coordinate) ** 2, axis=1))

        # select faces containing the closest vertex
        closest_faces = self.faces[np.any(self.faces == np.argmin(distance_vector), axis=1)]

        # point of coordinate of plane formed by triangle, boolean whether this point is inside the triangle,
        # and the normal of the triangle
        face_projections, face_normals, point_on_face = [None, ] * len(closest_faces), \
                                                        [None, ] * len(closest_faces), \
                                                        [None, ] * len(closest_faces)

        for i, face in enumerate(closest_faces):
            # get triangle vertices
            v1, v2, v3 = self.verts[face[0]], self.verts[face[1]], self.verts[face[2]]

            # calculate triangle normal
            normal = Vector(np.cross(v2 - v1, v3 - v1), normalize=True).inverse()  # normal needs to point outwards

            # get particle coordinate projected on triangle face and bool telling whether its
            face_projections[i], point_on_face[i], face_normals[i] = \
                *point_3d_in_triangle(coordinate, v1, v2, v3), normal

        sigma = find_std(face_normals)  # standard deviation of normals

        # consider cases of where the coordinate is relative to the surface
        if sum(point_on_face) == 1:  # this is the easy case, just above a single triangle

            membrane_normal, membrane_point = face_normals[point_on_face.index(True)], \
                                              face_projections[point_on_face.index(True)]

        elif sum(point_on_face) > 1:  # above two or more, select triangle with shortest distance

            # get the id of the closest face
            id, _ = min([(i, distance.euclidean(p, coordinate)) for i, (p, t)
                         in enumerate(zip(face_projections, point_on_face)) if t], key=lambda x: x[1])
            membrane_normal, membrane_point = face_normals[id], face_projections[id]

        else:  # not above any triangle, it is either above an edge or above a vert

            # get all edges connected to closest vert
            edges = faces_to_edges(closest_faces, np.argmin(distance_vector))

            # get edge projections, edge normals, and whether the projected coordinate is on the line segment
            # (see above, similar to faces)
            edge_projections, edge_normals, point_on_edge = [None, ] * len(edges), \
                                                            [None, ] * len(edges), \
                                                            [None, ] * len(edges)

            for i, edge in enumerate(edges):
                edge_projections[i], point_on_edge[i], edge_normals[i] = \
                    *point_3d_in_line(coordinate, self.verts[edge[0]], self.verts[edge[1]]), \
                    get_edge_vector(edge, closest_faces, self.verts)

            if sum(point_on_edge) == 1:  # easy case, above single edge

                membrane_normal, membrane_point = edge_normals[point_on_edge.index(True)], \
                                                  edge_projections[point_on_edge.index(True)]

            elif sum(point_on_edge) > 1:  # above two or more edges

                id, _ = min([(i, distance.euclidean(p, coordinate)) for i, (p, t)
                             in enumerate(zip(edge_projections, point_on_edge)) if t], key=lambda x: x[1])
                membrane_normal, membrane_point = edge_normals[id], edge_projections[id]

            else:  # finally, if not above anything else, select the vertex

                membrane_normal = Vector(self.normals[np.argmin(distance_vector)])
                membrane_point = self.verts[np.argmin(distance_vector)]

        return membrane_point, membrane_normal, sigma

    def find_particle_orientations(self, coordinates, rotations, verbose=False):

        n_particles = coordinates.shape[0]
        
        # initialize some lists
        distances, angle_stds, final_zxz, particle_arrows, membrane_arrows = \
            [None, ] * n_particles, [None, ] * n_particles, [None, ] * n_particles, \
            [None, ] * n_particles, [None, ] * n_particles

        # exclusion_count = 0
        for i in range(n_particles):

            coordinate = coordinates[i]
            zxz_ref_to_par = rotations[i]
            # zxz_ref_to_par = p.getRotation().toVector(convention='zxz')  # z1, x, z2

            # find closest point on membrane
            membrane_point, membrane_normal, angular_variation = self.find_membrane_surface(coordinate)

            if verbose:
                print('angular variation of membrane normals: ', angular_variation)

            # get rotation matrix and convert to axis-angle
            # rot_par_to_ref = transform.Rotation.from_euler('ZXZ', zxz_ref_to_par, degrees=True).as_matrix()
            rot_par_to_ref = rotation_matrix(rotation=zxz_ref_to_par,
                                             rotation_order='rzxz')[:3, :3].T  # transpose for ref to par

            if self.reference_unit_vector is not None:
                pass  # add additional rotation to particle

            # get rot of membrane surface to z axis unit vec
            rot_mem_to_ref = self.z_axis_unit_vector.get_rotation(membrane_normal)

            # 'subtract' reference rotation from the particle rotation
            rot_par_to_mem = np.dot(rot_mem_to_ref, rot_par_to_ref)

            # convert to zxz, but invert the matrix to accomodate to pytom convention
            # zxz_in_mem = transform.Rotation.from_matrix().as_euler('ZXZ', degrees=True)
            zxz_in_mem = matToZXZ(np.linalg.inv(rot_par_to_mem))
            # zxz_in_mem = matToZXZ(np.linalg.inv(rot_par_to_mem)).toVector(convention='zxz')

            # create arrows for bild file
            particle_arrows[i] = (coordinate, self.z_axis_unit_vector.rotate(rot_par_to_ref).get())
            membrane_arrows[i] = (membrane_point, membrane_normal.get())
            distances[i], angle_stds[i], final_zxz[i] = \
                distance.euclidean(membrane_point, coordinate), angular_variation, zxz_in_mem

        return distances, angle_stds, particle_arrows, membrane_arrows, final_zxz
