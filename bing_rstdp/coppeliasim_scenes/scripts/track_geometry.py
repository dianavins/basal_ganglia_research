#!/usr/bin/env python3
"""
Track Geometry Generator for Bing 2020 Lane-Keeping Scenario
Generates parametric track coordinates based on paper specifications.

Track specifications from Bing et al. 2020:
- r1_inner = 1.75m, r1_outer = 2.25m
- r2_inner = 3.25m, r2_outer = 2.75m
- l1 = 5.0m (straight sections)
- Lane width = 0.5m
- 6 sections: A(straight), B(left), C(straight), D(left), E(right), F(left)
"""

import numpy as np
import json


class TrackGeometry:
    """Calculate parametric track geometry for Scenario 1"""

    def __init__(self):
        # Track dimensions from paper (Fig. 3)
        self.r1_inner = 1.75  # m
        self.r1_outer = 2.25  # m
        self.r2_inner = 3.25  # m
        self.r2_outer = 2.75  # m
        self.l1 = 5.0  # m (straight section length)
        self.lane_width = 0.5  # m

        # Calculate center lane radii
        self.r1_center = (self.r1_inner + self.r1_outer) / 2  # 2.0m
        self.r2_center = (self.r2_inner + self.r2_outer) / 2  # 3.0m

        # Dashed line parameters
        self.dash_length = 0.5  # m
        self.dash_gap = 0.3  # m
        self.line_width = 0.1  # m

    def generate_straight_section(self, start_pos, end_pos, num_points=50):
        """Generate points along a straight section"""
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            points.append([x, y, 0.0])
        return points

    def generate_arc_section(self, center, radius, start_angle, end_angle, num_points=50):
        """Generate points along a circular arc

        Args:
            center: [x, y] center of arc
            radius: radius of arc
            start_angle: starting angle in radians
            end_angle: ending angle in radians
            num_points: number of points to generate
        """
        points = []
        angles = np.linspace(start_angle, end_angle, num_points)
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y, 0.0])
        return points

    def generate_outer_lane_centerline(self):
        """Generate the complete outer lane centerline path"""
        points = []

        # Section A: Straight (starting at origin, going right)
        # Start: (0, 0), End: (l1, 0)
        section_a = self.generate_straight_section([0, 0], [self.l1, 0])
        points.extend(section_a)

        # Section B: Left turn with r1_center
        # Arc center: (l1, r1_center)
        # Start angle: -π/2, End angle: 0
        center_b = [self.l1, self.r1_center]
        section_b = self.generate_arc_section(center_b, self.r1_center, -np.pi/2, 0)
        points.extend(section_b)

        # Section C: Straight (going up)
        # Start: (l1 + r1_center, r1_center), End: (l1 + r1_center, r1_center + l1)
        start_c = [self.l1 + self.r1_center, self.r1_center]
        end_c = [self.l1 + self.r1_center, self.r1_center + self.l1]
        section_c = self.generate_straight_section(start_c, end_c)
        points.extend(section_c)

        # Section D: Left turn with r2_center
        # Arc center: (l1 + r1_center - r2_center, r1_center + l1)
        # Start angle: 0, End angle: π/2
        center_d = [self.l1 + self.r1_center - self.r2_center, self.r1_center + self.l1]
        section_d = self.generate_arc_section(center_d, self.r2_center, 0, np.pi/2)
        points.extend(section_d)

        # Section E: Right turn
        # This connects back towards the starting position
        # Arc center: (l1 + r1_center - r2_center, r1_center + l1 + r2_center)
        # Start angle: -π/2, End angle: 0
        center_e = [self.l1 + self.r1_center - self.r2_center, self.r1_center + self.l1 + self.r2_center]
        section_e = self.generate_arc_section(center_e, self.r1_center, -np.pi/2, 0)
        points.extend(section_e)

        # Section F: Left turn back to start
        # Arc center needs to close the loop back to origin
        # This is approximate - will need adjustment to close properly
        center_f = [self.r1_center, self.r1_center + self.l1 + self.r2_center]
        section_f = self.generate_arc_section(center_f, self.r1_center, 0, np.pi)
        points.extend(section_f)

        return points

    def offset_path(self, centerline, offset_distance):
        """Offset a path by a given distance (positive = right, negative = left)

        This uses normal vectors perpendicular to the path direction.
        """
        offset_points = []
        n = len(centerline)

        for i in range(n):
            # Calculate tangent direction
            if i == 0:
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
            elif i == n-1:
                dx = centerline[i][0] - centerline[i-1][0]
                dy = centerline[i][1] - centerline[i-1][1]
            else:
                dx = centerline[i+1][0] - centerline[i-1][0]
                dy = centerline[i+1][1] - centerline[i-1][1]

            # Normalize
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length

            # Normal vector (perpendicular, pointing right)
            nx = -dy
            ny = dx

            # Offset point
            x_offset = centerline[i][0] + nx * offset_distance
            y_offset = centerline[i][1] + ny * offset_distance
            offset_points.append([x_offset, y_offset, 0.0])

        return offset_points

    def generate_dashed_line_segments(self, centerline):
        """Convert centerline to dashed line segments"""
        segments = []
        current_distance = 0
        in_dash = True
        dash_start_idx = 0

        for i in range(1, len(centerline)):
            # Calculate distance along path
            dx = centerline[i][0] - centerline[i-1][0]
            dy = centerline[i][1] - centerline[i-1][1]
            segment_length = np.sqrt(dx**2 + dy**2)
            current_distance += segment_length

            # Check if we should switch between dash and gap
            threshold = self.dash_length if in_dash else (self.dash_length + self.dash_gap)

            if current_distance >= threshold:
                if in_dash:
                    # Save this dash segment
                    segments.append(centerline[dash_start_idx:i+1])
                # Switch state
                in_dash = not in_dash
                dash_start_idx = i
                current_distance = 0

        return segments

    def export_to_json(self, filename):
        """Export track geometry to JSON file for CoppeliaSim"""
        # Generate center lane
        outer_center = self.generate_outer_lane_centerline()

        # Generate lane boundaries
        outer_boundary = self.offset_path(outer_center, self.lane_width / 2)
        inner_boundary = self.offset_path(outer_center, -self.lane_width / 2)

        # Generate dashed center line
        dashed_segments = self.generate_dashed_line_segments(outer_center)

        # Inner lane (just offset further)
        inner_center = self.offset_path(outer_center, -self.lane_width)
        inner_outer_boundary = inner_boundary  # Shared boundary
        inner_inner_boundary = self.offset_path(inner_center, -self.lane_width / 2)

        data = {
            "track_parameters": {
                "r1_inner": self.r1_inner,
                "r1_outer": self.r1_outer,
                "r2_inner": self.r2_inner,
                "r2_outer": self.r2_outer,
                "l1": self.l1,
                "lane_width": self.lane_width
            },
            "outer_lane": {
                "centerline": outer_center,
                "outer_boundary": outer_boundary,
                "inner_boundary": inner_boundary,
                "center_dashed_segments": dashed_segments
            },
            "inner_lane": {
                "centerline": inner_center,
                "outer_boundary": inner_outer_boundary,
                "inner_boundary": inner_inner_boundary
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Track geometry exported to {filename}")
        print(f"Outer lane centerline points: {len(outer_center)}")
        print(f"Dashed line segments: {len(dashed_segments)}")


if __name__ == "__main__":
    track = TrackGeometry()
    output_file = "/mnt/c/Users/diana/basal_ganglia_research/bing_rstdp/coppeliasim_scenes/track_geometry.json"
    track.export_to_json(output_file)
