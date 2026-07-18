"""Crosswalk geometry shared by simulation-side detectors.

The rectangle calculations intentionally mirror ``Renderer.draw_crosswalks``.
Keeping the detector geometry independent of pygame also makes it usable by
headless policy evaluation and unit tests.
"""

from dataclasses import dataclass
import math


DIRECTIONS = ("north", "south", "east", "west")


@dataclass(frozen=True)
class AxisAlignedRectangle:
    """A floating-point rectangle in screen coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def corners(self):
        return (
            (self.left, self.top),
            (self.right, self.top),
            (self.right, self.bottom),
            (self.left, self.bottom),
        )


def direction_divider_width(config, direction):
    key = (
        "vertical_road_direction_divider_width"
        if direction in ("north", "south")
        else "horizontal_road_direction_divider_width"
    )
    return float(config[key])


def intersection_half_dimensions(config):
    """Return the same junction half-width/height used by the renderer."""
    lane_width = float(config["lane_width"])
    roads = config["roads"]
    vertical_width = max(
        (
            lane_width
            * (roads[direction]["incoming"] + roads[direction]["outgoing"])
            + direction_divider_width(config, direction)
            for direction in ("north", "south")
            if roads.get(direction, {}).get("enabled", False)
        ),
        default=0.0,
    )
    horizontal_width = max(
        (
            lane_width
            * (roads[direction]["incoming"] + roads[direction]["outgoing"])
            + direction_divider_width(config, direction)
            for direction in ("east", "west")
            if roads.get(direction, {}).get("enabled", False)
        ),
        default=0.0,
    )
    return vertical_width / 2.0, horizontal_width / 2.0


def crosswalk_rectangles(config):
    """Return physical marked-crosswalk rectangles for enabled approaches."""
    roads = config["roads"]
    lane_width = float(config["lane_width"])
    width = float(config["window"]["width"])
    height = float(config["window"]["height"])
    # Renderer uses integer screen centres even when a window dimension is odd.
    center_x = float(int(width) // 2)
    center_y = float(int(height) // 2)
    half_width, half_height = intersection_half_dimensions(config)
    setback = float(config["crosswalk_intersection_offset"])
    depth = float(config["crosswalk_width"])
    rectangles = {}

    for direction in DIRECTIONS:
        road = roads.get(direction, {})
        if not road.get("enabled", False):
            continue
        road_width = (
            lane_width * (road["incoming"] + road["outgoing"])
            + direction_divider_width(config, direction)
        )
        if direction == "north":
            rectangles[direction] = AxisAlignedRectangle(
                center_x - road_width / 2.0,
                center_y - half_height - setback - depth,
                road_width,
                depth,
            )
        elif direction == "south":
            rectangles[direction] = AxisAlignedRectangle(
                center_x - road_width / 2.0,
                center_y + half_height + setback,
                road_width,
                depth,
            )
        elif direction == "west":
            rectangles[direction] = AxisAlignedRectangle(
                center_x - half_width - setback - depth,
                center_y - road_width / 2.0,
                depth,
                road_width,
            )
        else:  # east
            rectangles[direction] = AxisAlignedRectangle(
                center_x + half_width + setback,
                center_y - road_width / 2.0,
                depth,
                road_width,
            )

    return rectangles


def _rect_value(rect, edge, fallback):
    value = getattr(rect, edge, None)
    if value is not None:
        return float(value)
    return float(fallback())


def _vehicle_polygon(vehicle):
    """Return the actual vehicle body footprint as a convex polygon."""
    get_corners = getattr(vehicle, "get_corners", None)
    if callable(get_corners):
        corners = get_corners()
        if corners and len(corners) >= 3:
            return tuple((float(x), float(y)) for x, y in corners)

    get_rect = getattr(vehicle, "get_rect", None)
    if not callable(get_rect):
        return ()
    rect = get_rect()
    left = _rect_value(rect, "left", lambda: rect.x)
    top = _rect_value(rect, "top", lambda: rect.y)
    right = _rect_value(rect, "right", lambda: rect.x + rect.width)
    bottom = _rect_value(rect, "bottom", lambda: rect.y + rect.height)
    return (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    )


def polygon_overlaps_rectangle(polygon, rectangle):
    """Return whether a convex polygon has positive-area rectangle overlap.

    The separating-axis test avoids false occupancy from the oversized square
    returned by ``Vehicle.get_rect`` while a vehicle is rotating.
    """
    if len(polygon) < 3 or rectangle.width <= 0.0 or rectangle.height <= 0.0:
        return False

    # Almost every vehicle is far from every given crosswalk. Reject those
    # pairs with a cheap bounding-box comparison before constructing SAT
    # projections; only turning vehicles near the marking need the full test.
    polygon_left = min(point[0] for point in polygon)
    polygon_right = max(point[0] for point in polygon)
    polygon_top = min(point[1] for point in polygon)
    polygon_bottom = max(point[1] for point in polygon)
    epsilon = 1e-9
    if (
        polygon_right <= rectangle.left + epsilon
        or rectangle.right <= polygon_left + epsilon
        or polygon_bottom <= rectangle.top + epsilon
        or rectangle.bottom <= polygon_top + epsilon
    ):
        return False
    if all(
        abs(
            polygon[(index + 1) % len(polygon)][0]
            - polygon[index][0]
        )
        <= epsilon
        or abs(
            polygon[(index + 1) % len(polygon)][1]
            - polygon[index][1]
        )
        <= epsilon
        for index in range(len(polygon))
    ):
        return True

    axes = [(1.0, 0.0), (0.0, 1.0)]
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        edge_x = next_point[0] - point[0]
        edge_y = next_point[1] - point[1]
        length = math.hypot(edge_x, edge_y)
        if length > 1e-12:
            axes.append((-edge_y / length, edge_x / length))

    rectangle_corners = rectangle.corners
    for axis_x, axis_y in axes:
        polygon_projection = [
            x * axis_x + y * axis_y for x, y in polygon
        ]
        rectangle_projection = [
            x * axis_x + y * axis_y for x, y in rectangle_corners
        ]
        if (
            max(polygon_projection) <= min(rectangle_projection) + epsilon
            or max(rectangle_projection) <= min(polygon_projection) + epsilon
        ):
            return False
    return True


def vehicle_overlaps_crosswalk(vehicle, rectangle):
    """Return whether a vehicle's body currently occupies a crosswalk."""
    return polygon_overlaps_rectangle(_vehicle_polygon(vehicle), rectangle)


def _point_to_segment_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, start)
    progress = max(
        0.0,
        min(
            1.0,
            (
                (point[0] - start[0]) * dx
                + (point[1] - start[1]) * dy
            )
            / length_squared,
        ),
    )
    closest = (
        start[0] + progress * dx,
        start[1] + progress * dy,
    )
    return math.dist(point, closest)


def _point_inside_polygon(point, polygon):
    """Return whether a point lies inside a convex or concave polygon."""
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def point_to_polygon_distance(point, polygon):
    """Return zero inside a polygon, otherwise distance to its boundary."""
    if len(polygon) < 3:
        return math.inf
    if _point_inside_polygon(point, polygon):
        return 0.0
    return min(
        _point_to_segment_distance(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def analyze_crosswalk_safety(
    vehicles,
    pedestrians,
    rectangles,
    pixels_per_meter,
    safety_margin_m=0.5,
):
    """Return vehicle occupancy and true near-conflict counts per crosswalk.

    Co-occupancy anywhere on a wide crosswalk is useful controller context,
    but it is not automatically a physical conflict: a vehicle and a
    pedestrian can be in different carriageways or lanes.  A conflict is
    counted only when the pedestrian circle is within the configured safety
    margin of an occupying vehicle's actual (possibly rotated) body polygon.
    """
    vehicle_counts = {crossing: 0 for crossing in rectangles}
    conflict_counts = {crossing: 0 for crossing in rectangles}
    margin_pixels = max(0.0, float(safety_margin_m)) * max(
        1e-9,
        float(pixels_per_meter),
    )
    active_pedestrians = {crossing: [] for crossing in rectangles}
    for pedestrian in pedestrians:
        crossing = getattr(pedestrian, "crossing", None)
        if crossing not in active_pedestrians:
            continue
        safely_waiting = getattr(pedestrian, "is_safely_waiting", None)
        if callable(safely_waiting):
            safely_waiting = bool(safely_waiting())
        else:
            safely_waiting = bool(
                getattr(pedestrian, "waiting", False)
                and not getattr(
                    pedestrian,
                    "has_reached_divider",
                    False,
                )
            )
        if safely_waiting:
            continue
        position = getattr(pedestrian, "position", None)
        if position is not None:
            active_pedestrians[crossing].append(pedestrian)

    for vehicle in vehicles:
        polygon = _vehicle_polygon(vehicle)
        if not polygon:
            continue
        for crossing, rectangle in rectangles.items():
            if not polygon_overlaps_rectangle(polygon, rectangle):
                continue
            vehicle_counts[crossing] += 1
            for pedestrian in active_pedestrians[crossing]:
                clearance = max(
                    0.0,
                    float(getattr(pedestrian, "radius", 0.0)),
                ) + margin_pixels
                if (
                    point_to_polygon_distance(pedestrian.position, polygon)
                    <= clearance
                ):
                    conflict_counts[crossing] += 1

    return vehicle_counts, conflict_counts


def count_crosswalk_vehicle_occupancies(vehicles, rectangles):
    """Count vehicle footprints in each crosswalk, computing each pose once."""
    counts = {crossing: 0 for crossing in rectangles}
    for vehicle in vehicles:
        polygon = _vehicle_polygon(vehicle)
        if not polygon:
            continue
        for crossing, rectangle in rectangles.items():
            if polygon_overlaps_rectangle(polygon, rectangle):
                counts[crossing] += 1
    return counts
