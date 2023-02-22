"""
SquidSofa
Simulation of a Soft Arm with FEM computed deformations.
The SOFA simulation contains two models of a Beam:
    * one to apply forces and compute deformations
    * one to apply the network predictions
"""

# Python related imports
import os
import sys
from numpy.random import randint, uniform
from time import sleep
from numpy.linalg import norm

# Sofa related imports
import Sofa.Simulation
import SofaRuntime

# DeepPhysX related imports
from DeepPhysX.Sofa.Environment.SofaEnvironment import SofaEnvironment

# Working session related imports
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from parameters import p_grid


class SquidSofa(SofaEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):

        SofaEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)


    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        """

        # Add required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])
        plugins = ['SofaCaribou', 'SofaBaseTopology', 'SofaGeneralEngine',
                   'SofaEngine', 'SofaOpenglVisual', 'SofaBoundaryCondition', 'SoftRobots',
                   'SofaSparseSolver','SofaPreconditioner','SofaPython3','SofaConstraint',
                   'SofaImplicitOdeSolver','SofaLoader','SofaSimpleFem',
                   "SofaDeformable", 'SofaGeneralLoader']
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showWireframe")

        # Create FEM of the squid arm
        self.createFEM()
        # Add cables to the arm
        self.addCables()
        # Add the effector at the tip
        #self.addEfector()

    def createFEM(self):
        armYoungModulus=110e-3
        armPoissonRatio=0.33
        armMass=0.1279
        ##########################################
        # FEM Model                              #
        ##########################################
        arm = self.root.addChild('arm')
        arm.addObject('EulerImplicitSolver', name='odesolver', firstOrder=True)
        arm.addObject('SparseLDLSolver', name='preconditioner', template='CompressedRowSparseMatrixMat3x3d')

        arm.addObject('MeshVTKLoader', name='loader', filename='./Environment/mesh/arm-final.vtk')
        arm.addObject('TetrahedronSetTopologyContainer', position='@loader.position', tetras='@loader.tetras', name='container')
        arm.addObject('TetrahedronSetTopologyModifier')

        # Add a mechanical object component to stores the DoFs of the model
        arm.addObject('MechanicalObject', name='tetras', template='Vec3')

        # Gives a mass to the model
        arm.addObject('UniformMass', totalMass=armMass)
        
        ## Implementing constitutive law of material and mass
        # Define material to be simulated by adding a ForceField component
        # This describes what internal forces are created when the object is deformed
        # Additionally, this will define how stiff or soft the material is as well as its behaviour

        arm.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=armPoissonRatio,  youngModulus=armYoungModulus)

        # To facilitate the selection of DoFs, SOFA has a concept called ROI (Region of Interest).
        # The idea is that ROI component "select" all DoFS that are enclosed by their "region".
        # We use ROI here to select a group of finger's DoFs that will be constrained to stay
        # at a fixed position.

        # The arm base is in the x-y plane and the height is in the z-direction
        # box points = [xmin, ymin, zmin, xmax, ymax, zmax]
        arm.addObject('BoxROI', name='ROI', box=[-40, -40, 0, 40, 40, 30], drawBoxes=True)

        # RestShapeSpringsForceField is one way in Sofa to implement fixed point constraint.
        # Here the constraints are applied to the DoFs selected by the previously defined BoxROI
        arm.addObject('RestShapeSpringsForceField', points='@ROI.indices', stiffness=1e12)

        arm.addObject('LinearSolverConstraintCorrection')

        ##########################################
        # Visualization                          #
        ##########################################
        # In Sofa, visualization is handled by adding a rendering model.
        # add an empty child node to store this rendering model.
        armVisu = arm.addChild('visu')
        armVisu.addObject('MeshSTLLoader', name='Loader', filename='./Environment/mesh/arm-final.stl')
        armVisu.addObject('OglModel', src='@Loader', color=[0.7, 0.7, 1])
        armVisu.addObject('BarycentricMapping')

    def addCables(self):
        arm = self.root.arm

        ######## Cable 1 ###########
        cable_1 = arm.addChild('cable_1')
        position =[]
        for i in range(601):
            position.append([-(i/30) + 25, 0, i])

        cable_1.addObject('MechanicalObject',  position=position)

        # Add a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last index is where the pullPoint is connected.
        # By default, the Cable is controlled by displacement, rather than force.
        cable_1.addObject('CableConstraint', name="aCable_1", indices=list(range(len(position))), pullPoint=[25, 0, -15])

        # This adds a BarycentricMapping. A BarycentricMapping is a key element as it will add a bidirectional link
        # between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        # to the finger and vice-versa;
        cable_1.addObject('BarycentricMapping')


        ######## Cable 2 ###########
        cable_2 = arm.addChild('cable_2')
        position =[]
        for i in range(601):
            position.append([ (i/30) - 25, 0, i]) # -10 in y

        cable_2.addObject('MechanicalObject',  position=position)

        # Add a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last index is where the pullPoint is connected.
        # By default, the Cable is controlled by displacement, rather than force.
        cable_2.addObject('CableConstraint', name="aCable_2", indices=list(range(len(position))), pullPoint=[-25, 0, -15])

        # This adds a BarycentricMapping. A BarycentricMapping is a key element as it will add a bidirectional link
        # between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        # to the finger and vice-versa;
        cable_2.addObject('BarycentricMapping')

        ######## Cable 3 ###########
        cable_3 = arm.addChild('cable_3')
        position =[]
        for i in range(601):
            position.append([0, -(i/30) + 25, i]) # -10 in x

        cable_3.addObject('MechanicalObject',  position=position)

        # Add a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last index is where the pullPoint is connected.
        # By default, the Cable is controlled by displacement, rather than force.
        cable_3.addObject('CableConstraint', name="aCable_3", indices=list(range(len(position))), pullPoint=[0, 25, -15])

        # This adds a BarycentricMapping. A BarycentricMapping is a key element as it will add a bidirectional link
        # between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        # to the finger and vice-versa;
        cable_3.addObject('BarycentricMapping')

        
        ######## Cable 4 ###########
        cable_4 = arm.addChild('cable_4')
        position =[]
        for i in range(601):
            position.append([0,(i/30) - 25, i])

        cable_4.addObject('MechanicalObject',  position=position)

        # Add a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last index is where the pullPoint is connected.
        # By default, the Cable is controlled by displacement, rather than force.
        cable_4.addObject('CableConstraint', name="aCable_4", indices=list(range(len(position))), pullPoint=[0, -25, -15])

        # This adds a BarycentricMapping. A BarycentricMapping is a key element as it will add a bidirectional link
        # between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        # to the finger and vice-versa;
        cable_4.addObject('BarycentricMapping')

    def addEffector(self):
        arm = self.root.arm
        # We will add an point to the end of the arm to act as end effector
        effectors = arm.addChild('Effectors')
        effectors.addObject('MechanicalObject', position=[0., 0., 600.], name="myEndEffector")
        effectors.addObject('BarycentricMapping', mapForces=False, mapMasses=False)
    

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        arm = self.root.arm
        self.cable_1 = arm.getChild('cable_1')
        self.cable_2 = arm.getChild('cable_2')
        self.cable_3 = arm.getChild('cable_3')
        self.cable_4 = arm.getChild('cable_4')

        # # To get the position of the end effector
        # self.Effector = arm.getChild('Effectors')
        # self.endEffector = self.Effector.myEndEffector
        # self.effectorPosition = self.endEffector.findData('position').value

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """
        inputvalue_c1 = self.cable_1.aCable_1
        inputvalue_c2 = self.cable_2.aCable_2
        inputvalue_c3 = self.cable_3.aCable_3
        inputvalue_c4 = self.cable_4.aCable_4 

        disp_1 = inputvalue_c1.findData('displacement').value 
        disp_2 = inputvalue_c2.findData('displacement').value
        disp_3 = inputvalue_c3.findData('displacement').value
        disp_4 = inputvalue_c4.findData('displacement').value

        disp_4 += 1
        inputvalue_c4.findData('displacement').value = disp_4 

        sleep(0.01 * randint(0, 10))


    # def onAnimateEndEvent(self, event):
    #     """
    #     Called within the Sofa pipeline at the end of the time step.
    #     """

    #     # Check whether if the solver diverged or not
    #     if not self.check_sample():
    #         print("Solver diverged.")

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # Check if the solver converged while computing FEM
        if self.create_model['fem']:
            if not self.solver.converged.value:
                # Reset simulation if solver diverged to avoid unwanted behaviour in following samples
                Sofa.Simulation.reset(self.root)
            return self.solver.converged.value
        return True

    def close(self):
        """
        Shutdown procedure.
        """

        print("Bye!")
