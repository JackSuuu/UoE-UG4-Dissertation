# Registered continuation baseline: released, seated placement

## Question and budget

Does the same local task-finetuned full OpenVLA complete placement if allowed to continue beyond LIBERO's permissive native success event? This is a development capability check, not physical-verifier efficacy or safety certification.

Reuse only initial indices0/1/2 from the preceding book/caddy pilot. Keep checkpoint revision `80970322773f81baa2e22fe495d0487b93a05cfa`, frozen full7B BF16, H1/k1, official preprocessing/unnormalization, seed0,20Hz OSC,10settling steps and a **total maximum520policy steps from episode start**. No extra520steps after success, corrective controller, scripted release or actor training. Three healthy GPU workers, batch1, detached `setsid nohup`. Preserve old artifacts. Repeat from original initial states; do not restore an incomplete controller state at the old termination.

Log native success/done as an event; do not terminate on it. LIBERO's returned done is overwritten with its native predicate; robosuite's internal time horizon is1000 and therefore remains beyond this bounded run. Stop on the registered completion dwell or at520steps. No acceptance-threshold changes or automatic reruns after outcomes. Roots remain development-only.

## Completion definition, frozen before new full rollouts

Evaluate the book's **collision box**, including local offset and current orientation, in caddy-body coordinates. The rear lower compartment is bounded by actual opposing box faces (geom5/4inX, geom2/7inY) and base geom1:

- Xbounds `[0.00247, 0.06277]`m; Ybounds `[-0.06184, 0.06699]`m; floorZ `0.01381`m, derived at runtime from compiled geometry.
- All eight book corners' XY coordinates lie within these bounds. The caddy is open at the top; no artificial upper-Z containment is required. Orientation affects the corners, rather than using an arbitrary target quaternion. Angled upper-wall sections remain represented in contact diagnostics, not approximated away by the rectangle.
- The lowest book corner is within5mm of the physical floor plane.
- No book/gripper contact in any of the25physics-step solver readouts in the control interval. This tests physical detachment, not merely an open-gripper command.
- Book/caddy contact at the interval's last solver readout, to distinguish a floating pose from seated support (combined with floor condition).
- Book/caddy penetration no more than1mm at any recorded solver substep of the interval.
- Book linear speed≤0.02m/s and angular speed≤0.1rad/s at the20Hz observation boundary.
- All conditions hold for20consecutive control intervals (1second), while the original VLA continues acting. First passing dwell ends the run. Record all component conditions and maximum streak, including unsuccessful episodes.

These5mm/1mm/speed/dwell tolerances are explicit **development operational definitions**, selected before continuation outcomes to test released seated placement, not established physical safety standards or a millimetre-precision performance claim. Horizontal box/floor tests are geometry-aware approximations, not a generic arbitrary-mesh containment proof. Geometry and speed are sampled at20Hz; gripper contact and penetration at500Hz. Do not describe this as continuously certified constraint satisfaction. CVR and safe success remain null until a separately justified force/impact and numerical-convergence protocol exists.

## Logging and attribution

Record all520steps or the complete dwell, not just a short post-trigger segment. Persist video, initial/policy/final frames, native tokens/actions, native success, each geometric/release/stability condition, book-corner coordinates, speeds, solver contact readouts for book/caddy-related pairs at every physics substep, and simulator state arrays. Force values are diagnostics, never inputs to the image-only actor. `ContactRecorder` wraps existing `sim.step` calls, adds no forward/step/control calls and changes no solver settings. It reads contacts/forces associated with that solver step; these can differ from post-integration geometric quantities by a timestep.

Check repeat-run action/state agreement through the old first-native-success prefix before interpreting continuation. Attribute failure components from recorded conditions and videos; never equate a contact count with a safety violation. If none complete, inspect whether the failure is a task-capability, decoding/control-compatibility, geometry-definition or numerical issue before designing learned intervention.

## Pre-launch checks

Three new acceptance checks reject (a) reference-point-only containment, (b) grasped/airborne poses and (c) misaligned/rotating placements; a seated detached case passes. Two existing simulator ABI checks pass. Replayed all185original root0actions with instrumentation and metrics enabled: all186saved states matched exactly (max absolute error0), first-native step184unchanged,25solver samples/control interval. One additional interface step confirmed that execution can continue after native done. This diagnostic replay is not a VLA continuation result; output `/home/ami/scratch/libero_continuation_replay_check.json`.

Protocol/code are committed and pushed before the three full continuation runs. Exact-simulator intervention and learned training remain subsequent decisions contingent on complete-task feasibility.
