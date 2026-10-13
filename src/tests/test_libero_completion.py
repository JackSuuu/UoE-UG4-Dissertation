import unittest
import numpy as np
from adapters.libero_completion import box_corners, placement_conditions


class CompletionTests(unittest.TestCase):
    def checks(self, center=(0.03, 0, 0.08), rotation=None, **changes):
        corners = box_corners(center, np.eye(3) if rotation is None else rotation,
                              np.array([0.014, 0.055, 0.067]))
        args = dict(corners=corners, bounds=np.array([[0, -0.065], [0.065, 0.065]]),
                    floor=0.013, linear_speed=0., angular_speed=0.,
                    gripper_contact=False, caddy_contact=True, penetration=0.)
        args.update(changes)
        return placement_conditions(**args)

    def test_reference_point_inside_but_geometry_outside(self):
        self.assertFalse(self.checks(center=(0.002, 0, 0.08))['footprint_inside'])

    def test_grasped_or_airborne_is_not_released_placement(self):
        self.assertFalse(all(self.checks(gripper_contact=True).values()))
        self.assertFalse(all(self.checks(center=(0.03, 0, 0.10), caddy_contact=False).values()))

    def test_rotation_and_motion_matter(self):
        angle = np.pi / 4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        self.assertFalse(self.checks(rotation=rotation)['footprint_inside'])
        self.assertFalse(all(self.checks(angular_speed=0.2).values()))
        self.assertTrue(all(self.checks().values()))


if __name__ == '__main__':
    unittest.main()
