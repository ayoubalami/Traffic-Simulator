import unittest
from types import SimpleNamespace

from config import CONFIG, build_runtime_config
from simulation.crosswalk_geometry import (
    AxisAlignedRectangle,
    analyze_crosswalk_safety,
    crosswalk_rectangles,
    vehicle_overlaps_crosswalk,
)
from simulation.simulation import Simulation


class DetectorVehicle:
    def __init__(self, rectangle, *, corners=None):
        self._rectangle = rectangle
        self._corners = corners
        self.cleared_intersection = False
        self.road_direction = "north"
        self.current_speed = 10.0
        self.speed = 20.0
        self.is_emergency = False
        self.stopped = False
        self.turning = False
        self.is_turning_vehicle = False
        self.has_turned = False
        self.turn_side = None

    def get_rect(self):
        return self._rectangle

    def get_corners(self):
        return self._corners


class CrosswalkObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)

    def test_crosswalk_rectangles_match_rendered_geometry(self):
        rectangles = crosswalk_rectangles(self.config)
        lane_width = self.config["lane_width"]
        vertical_road_width = (
            lane_width
            * (
                self.config["roads"]["north"]["incoming"]
                + self.config["roads"]["north"]["outgoing"]
            )
            + self.config["vertical_road_direction_divider_width"]
        )
        horizontal_road_width = (
            lane_width
            * (
                self.config["roads"]["west"]["incoming"]
                + self.config["roads"]["west"]["outgoing"]
            )
            + self.config["horizontal_road_direction_divider_width"]
        )
        center_x = self.config["window"]["width"] // 2
        center_y = self.config["window"]["height"] // 2
        depth = self.config["crosswalk_width"]
        setback = self.config["crosswalk_intersection_offset"]

        self.assertEqual(rectangles["north"].width, vertical_road_width)
        self.assertEqual(rectangles["north"].height, depth)
        self.assertEqual(
            rectangles["north"].y,
            center_y - horizontal_road_width / 2 - setback - depth,
        )
        self.assertEqual(
            rectangles["south"].y,
            center_y + horizontal_road_width / 2 + setback,
        )
        self.assertEqual(rectangles["west"].height, horizontal_road_width)
        self.assertEqual(
            rectangles["west"].x,
            center_x - vertical_road_width / 2 - setback - depth,
        )
        self.assertEqual(
            rectangles["east"].x,
            center_x + vertical_road_width / 2 + setback,
        )

    def test_rotated_vehicle_uses_body_polygon_not_bounding_square(self):
        crosswalk = AxisAlignedRectangle(0.0, 0.0, 2.0, 2.0)
        bounding_square = SimpleNamespace(
            left=1.0,
            top=1.0,
            right=4.0,
            bottom=4.0,
        )
        vehicle = DetectorVehicle(
            bounding_square,
            corners=((1.9, 3.0), (3.0, 1.9), (4.0, 3.0), (3.0, 4.0)),
        )

        self.assertFalse(vehicle_overlaps_crosswalk(vehicle, crosswalk))

        vehicle._corners = ((1.0, 2.0), (2.0, 1.0), (3.0, 2.0), (2.0, 3.0))
        self.assertTrue(vehicle_overlaps_crosswalk(vehicle, crosswalk))

    def test_wide_crosswalk_cooccupancy_is_not_automatically_a_conflict(self):
        crosswalk = AxisAlignedRectangle(0.0, 0.0, 100.0, 20.0)
        vehicle = DetectorVehicle(
            SimpleNamespace(left=80.0, top=2.0, right=95.0, bottom=18.0)
        )
        pedestrian = SimpleNamespace(
            crossing="north",
            position=(10.0, 10.0),
            radius=2.0,
            is_safely_waiting=lambda: False,
        )

        occupancy, conflicts = analyze_crosswalk_safety(
            [vehicle],
            [pedestrian],
            {"north": crosswalk},
            pixels_per_meter=10.0,
            safety_margin_m=0.5,
        )

        self.assertEqual(occupancy["north"], 1)
        self.assertEqual(conflicts["north"], 0)

        pedestrian.position = (77.0, 10.0)
        _, conflicts = analyze_crosswalk_safety(
            [vehicle],
            [pedestrian],
            {"north": crosswalk},
            pixels_per_meter=10.0,
            safety_margin_m=0.5,
        )
        self.assertEqual(conflicts["north"], 1)

    def test_observation_separates_waiters_crossers_and_vehicle_occupancy(self):
        simulation = Simulation(self.config, random_seed=1)
        north_crosswalk = crosswalk_rectangles(self.config)["north"]
        vehicle_rect = SimpleNamespace(
            left=north_crosswalk.left + 5.0,
            top=north_crosswalk.top + 2.0,
            right=north_crosswalk.left + 15.0,
            bottom=north_crosswalk.bottom - 2.0,
            center=(
                north_crosswalk.left + 10.0,
                (north_crosswalk.top + north_crosswalk.bottom) / 2.0,
            ),
        )
        simulation.vehicles = [DetectorVehicle(vehicle_rect)]
        simulation.pedestrians = [
            SimpleNamespace(
                crossing="north",
                waiting=True,
                has_reached_divider=False,
                is_safely_waiting=lambda: True,
            ),
            SimpleNamespace(
                crossing="north",
                waiting=False,
                has_reached_divider=False,
                is_safely_waiting=lambda: False,
            ),
            SimpleNamespace(
                crossing="south",
                waiting=True,
                has_reached_divider=True,
                is_safely_waiting=lambda: True,
            ),
        ]

        observation = simulation.get_signal_observation("ns")

        self.assertEqual(observation["waiting_pedestrian_counts"]["north"], 1)
        self.assertEqual(
            observation["active_crossing_pedestrian_counts"]["north"],
            1,
        )
        self.assertEqual(
            observation["active_crossing_pedestrian_counts"]["south"],
            0,
        )
        self.assertEqual(
            observation["crosswalk_vehicle_occupancy_counts"]["north"],
            1,
        )
        self.assertEqual(
            observation[
                "vehicle_pedestrian_crosswalk_conflict_counts"
            ]["north"],
            0,
        )
        for crossing in ("south", "east", "west"):
            self.assertEqual(
                observation["crosswalk_vehicle_occupancy_counts"][crossing],
                0,
            )

    def test_walk_waits_for_committed_vehicle_approaching_exit_crosswalk(self):
        simulation = Simulation(
            self.config,
            random_seed=1,
            movement_score_provider=lambda observation: {},
        )
        north = simulation._crosswalk_rectangles["north"]
        approaching_rect = SimpleNamespace(
            left=north.right - 20.0,
            right=north.right - 5.0,
            top=north.bottom + 5.0,
            bottom=north.bottom + 25.0,
            center=(north.right - 12.5, north.bottom + 15.0),
        )
        vehicle = DetectorVehicle(approaching_rect)
        vehicle.road_direction = "south"
        vehicle.cleared_intersection = True
        simulation.vehicles = [vehicle]

        self.assertFalse(simulation._can_start_pedestrian_walk("north"))

        vehicle._rectangle = SimpleNamespace(
            left=north.right - 20.0,
            right=north.right - 5.0,
            top=north.top - 25.0,
            bottom=north.top - 5.0,
            center=(north.right - 12.5, north.top - 15.0),
        )
        self.assertTrue(simulation._can_start_pedestrian_walk("north"))


if __name__ == "__main__":
    unittest.main()
