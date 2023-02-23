"""
SquidSofa
Simulation of a Soft Arm with FEM computed deformations.
The SOFA simulation contains two models of a Beam:
    * one to apply forces and compute deformations
    * one to apply the network predictions
"""

# Python related imports
import os

# Sofa related imports
import SofaRuntime

# DeepPhysX related imports
from DeepPhysX.Sofa.Environment.SofaEnvironment import SofaEnvironment


class SquidSofa(SofaEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):

        SofaEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

        self.cables = []
        self.effector = None

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        """

        # Add required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        plugins = ['Sofa.Component.ODESolver', 'Sofa.Component.LinearSolver', 'Sofa.Component.IO.Mesh',
                   'Sofa.Component.Mass', 'Sofa.Component.SolidMechanics.FEM.Elastic',
                   'Sofa.Component.Engine.Select', 'Sofa.Component.Constraint.Lagrangian.Correction',
                   'Sofa.GL.Component.Rendering3D', 'SoftRobots']
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showWireframe")

        # Create FEM of the squid arm
        self.createFEM()
        # Add cables to the arm
        self.addCables()
        # Add the effector at the tip
        self.addEffector()

    def createFEM(self):

        arm_young_modulus = 110e-3
        arm_poisson_ratio = 0.33
        arm_mass = 0.1279

        ##########################################
        # FEM Model                              #
        ##########################################
        self.root.addChild('arm')
        self.root.arm.addObject('EulerImplicitSolver', name='ODESolver', firstOrder=True)
        self.root.arm.addObject('SparseLDLSolver', name='Preconditioner', template='CompressedRowSparseMatrixMat3x3d')

        self.root.arm.addObject('MeshVTKLoader', name='Mesh', filename='./Environment/mesh/arm-final.vtk')
        self.root.arm.addObject('TetrahedronSetTopologyContainer', position='@Mesh.position', tetras='@Mesh.tetras',
                                name='container')
        self.root.arm.addObject('TetrahedronSetTopologyModifier')

        # Add a mechanical object component to stores the DoFs of the model
        self.root.arm.addObject('MechanicalObject', name='Tetras', template='Vec3')

        # Gives a mass to the model
        self.root.arm.addObject('UniformMass', totalMass=arm_mass)

        # Implementing constitutive law of material and mass
        # Define material to be simulated by adding a ForceField component
        # This describes what internal forces are created when the object is deformed
        # Additionally, this will define how stiff or soft the material is as well as its behaviour
        self.root.arm.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large',
                                poissonRatio=arm_poisson_ratio, youngModulus=arm_young_modulus)

        # To facilitate the selection of DoFs, SOFA has a concept called ROI (Region of Interest).
        # The idea is that ROI component "select" all DoFS that are enclosed by their "region".
        # We use ROI here to select a group of finger's DoFs that will be constrained to stay
        # at a fixed position.
        # The arm base is in the x-y plane and the height is in the z-direction
        # box points = [xmin, ymin, zmin, xmax, ymax, zmax]
        self.root.arm.addObject('BoxROI', name='ROI', box=[-40, -40, 0, 40, 40, 30], drawBoxes=True)

        # RestShapeSpringsForceField is one way in Sofa to implement fixed point constraint.
        # Here the constraints are applied to the DoFs selected by the previously defined BoxROI
        self.root.arm.addObject('RestShapeSpringsForceField', points='@ROI.indices', stiffness=1e12)
        self.root.arm.addObject('LinearSolverConstraintCorrection')

        ##########################################
        # Visualization                          #
        ##########################################
        # In Sofa, visualization is handled by adding a rendering model.
        # add an empty child node to store this rendering model.
        self.root.arm.addChild('visu')
        self.root.arm.visu.addObject('MeshSTLLoader', name='Mesh', filename='./Environment/mesh/arm-final.stl')
        self.root.arm.visu.addObject('OglModel', src='@Mesh', color=[0.7, 0.7, 1])
        self.root.arm.visu.addObject('BarycentricMapping')

    def addCables(self):

        # Cable 1
        self.root.arm.addChild('cable_1')
        position = [[-(i / 30) + 25, 0, i] for i in range(601)]
        self.root.arm.cable_1.addObject('MechanicalObject', position=position)

        # Add a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last index is where the pullPoint is connected.
        # By default, the Cable is controlled by displacement, rather than force.
        cable = self.root.arm.cable_1.addObject('CableConstraint', name="aCable_1", indices=list(range(len(position))),
                                                pullPoint=[25, 0, -15])
        self.cables.append(cable)
        # This adds a BarycentricMapping. A BarycentricMapping is a key element as it will add a bidirectional link
        # between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        # to the finger and vice-versa;
        self.root.arm.cable_1.addObject('BarycentricMapping')

        # Cable 2
        self.root.arm.addChild('cable_2')
        position = [[(i / 30) - 25, 0, i] for i in range(601)]
        self.root.arm.cable_2.addObject('MechanicalObject', position=position)
        cable = self.root.arm.cable_2.addObject('CableConstraint', name="aCable_2", indices=list(range(len(position))),
                                                pullPoint=[-25, 0, -15])
        self.cables.append(cable)
        self.root.arm.cable_2.addObject('BarycentricMapping')

        # Cable 3
        self.root.arm.addChild('cable_3')
        position = [[0, -(i / 30) + 25, i] for i in range(601)]
        self.root.arm.cable_3.addObject('MechanicalObject', position=position)
        cable = self.root.arm.cable_3.addObject('CableConstraint', name="aCable_3", indices=list(range(len(position))),
                                                pullPoint=[0, 25, -15])
        self.cables.append(cable)
        self.root.arm.cable_3.addObject('BarycentricMapping')

        # Cable 4
        self.root.arm.addChild('cable_4')
        position = [[0, (i / 30) - 25, i] for i in range(601)]
        self.root.arm.cable_4.addObject('MechanicalObject', position=position)
        cable = self.root.arm.cable_4.addObject('CableConstraint', name="aCable_4", indices=list(range(len(position))),
                                                pullPoint=[0, -25, -15])
        self.cables.append(cable)
        self.root.arm.cable_4.addObject('BarycentricMapping')

    def addEffector(self):

        # We will add a point to the end of the arm to act as end effector
        self.root.arm.addChild('effectors')
        self.effector = self.root.arm.effectors.addObject('MechanicalObject', position=[0., 0., 600.], name="Tip")
        self.root.arm.effectors.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        cable = self.cables[-1]
        cable.displacement.value += 1
        print("End effector position = ", self.effector.position.value[0])
