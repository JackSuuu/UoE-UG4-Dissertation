"""Causal diagnostic cues from logged geometry/contact; privileged, not deployed.

Only current/past measurements enter update. These rules localize failure phases;
they do not predict future contact or certify task safety.
"""
import numpy as np


class ExecutionDiagnostics:
    def __init__(self, bounds, floor):
        self.bounds = np.asarray(bounds)
        self.floor = float(floor)
        self.grasp_contact_seen = False
        self.placement_seen = False
        self.stalled_intervals = 0

    def update(self, row):
        c = row['conditions']
        self.grasp_contact_seen |= not c['released']
        recontact = self.placement_seen and not c['released']
        self.placement_seen |= row['candidate']
        stall = (self.grasp_contact_seen and not c['released'] and
                 c['supported'] and not c['footprint_inside'] and c['slow'])
        self.stalled_intervals = self.stalled_intervals + 1 if stall else 0
        corners = np.asarray(row['book_corners_caddy'])
        clearance = min(float(np.min(corners[:, :2] - self.bounds[0])),
                        float(np.min(self.bounds[1] - corners[:, :2])))
        if row['complete']:
            phase = 'released placement: dwell met'
        elif recontact:
            phase = 're-contact after valid placement'
        elif row['candidate']:
            phase = 'placement valid: accumulating dwell'
        elif self.stalled_intervals >= 20:
            phase = 'persistent contact / stalled alignment'
        elif self.grasp_contact_seen:
            phase = 'transport / align / place'
        else:
            phase = 'approach / grasp'
        return dict(phase=phase, signed_footprint_margin_mm=1000 * clearance,
                    bottom_floor_error_mm=1000 * (float(corners[:, 2].min()) - self.floor),
                    recontact_after_placement=bool(recontact),
                    placement_preservation_opportunity=bool(row['candidate'] and not row['complete']),
                    contact_stall=self.stalled_intervals >= 20,
                    stall_intervals=self.stalled_intervals)
