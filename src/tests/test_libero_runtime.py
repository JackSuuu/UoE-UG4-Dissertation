"""Checks for the simulator ABI adaptations, independent of task success."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import unittest
import mujoco
import numpy as np
from adapters.libero_runtime import patch_joint_addresses, patch_mass_matrix


class LiberoCompatibilityTest(unittest.TestCase):
    def test_all_joint_address_types(self):
        from robosuite.utils.binding_utils import MjModel
        patch_joint_addresses()
        xml = '''<mujoco><worldbody>
        <body><freejoint name="free"/><geom size=".1"/></body>
        <body><joint name="ball" type="ball"/><geom size=".1"/></body>
        <body><joint name="slide" type="slide"/><geom size=".1"/></body>
        <body><joint name="hinge" type="hinge"/><geom size=".1"/></body>
        </worldbody></mujoco>'''
        model = MjModel(mujoco.MjModel.from_xml_string(xml))
        for name, qpos, qvel in [('free', (0, 7), (0, 6)), ('ball', (7, 11), (6, 9)),
                                 ('slide', 11, 9), ('hinge', 12, 10)]:
            self.assertEqual(model.get_joint_qpos_addr(name), qpos)
            self.assertEqual(model.get_joint_qvel_addr(name), qvel)

    def test_controller_mass_matrix_matches_native(self):
        from robosuite.controllers.base_controller import Controller
        from robosuite.utils.binding_utils import MjSim
        patch_joint_addresses()
        patch_mass_matrix()
        sim = MjSim.from_xml_string('''<mujoco><worldbody>
          <body><joint name="j"/><geom size=".1"/><site name="eef" pos=".1 0 0"/></body>
          </worldbody></mujoco>''')
        class ConcreteController(Controller):
            def run_controller(self):
                raise NotImplementedError

        controller = ConcreteController(sim, 'eef', dict(joints=[0], qpos=[0], qvel=[0]), ([-1], [1]))
        expected = np.zeros((sim.model.nv, sim.model.nv))
        mujoco.mj_fullM(sim.model._model, sim.data._data, expected)
        np.testing.assert_allclose(controller.mass_matrix, expected, atol=0, rtol=0)
        self.assertGreater(np.linalg.eigvalsh(controller.mass_matrix).min(), 0)


if __name__ == '__main__':
    unittest.main()
