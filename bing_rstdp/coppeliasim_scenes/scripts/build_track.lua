-- Track Builder for CoppeliaSim - Bing et al. 2018 Lane Keeping Task
-- Implements Scenario 1: Circular track with 2 lanes and 6 sections
-- Based on "End to End Learning of Spiking Neural Network based on R-STDP
-- for a Lane Keeping Vehicle" (Bing et al., 2018)

-- Track dimensions from paper (Figure 3a)
local TRACK_PARAMS = {
    -- Radii for sections B, D, F (left turns with r1)
    r1_inner = 1.75,  -- Inner boundary radius (m)
    r1_outer = 2.25,  -- Outer boundary radius (m)

    -- Radii for section E (right turn with r2)
    r2_inner = 3.25,  -- Inner boundary radius (m) - larger because right turn
    r2_outer = 2.75,  -- Outer boundary radius (m)

    -- Straight section length
    l1 = 5.0,  -- Length of straight sections A and C (m)

    -- Lane markings
    line_width = 0.05,      -- 5 cm wide lines (skinnier)
    line_height = 0.002,    -- 2 mm height
    dash_length = 0.5,      -- 50 cm dash segments
    dash_gap = 0.5,         -- 50 cm gaps between dashes

    -- Ground plane
    ground_size = 30.0,     -- 30m x 30m ground
}

-- Calculate center line radii (middle of the lane)
TRACK_PARAMS.r1_center = (TRACK_PARAMS.r1_inner + TRACK_PARAMS.r1_outer) / 2  -- = 2.0m
TRACK_PARAMS.r2_center = (TRACK_PARAMS.r2_inner + TRACK_PARAMS.r2_outer) / 2  -- = 3.0m
TRACK_PARAMS.lane_width = TRACK_PARAMS.r1_outer - TRACK_PARAMS.r1_inner       -- = 0.5m

-- Helper: Create a rectangular shape (for line segments)
function createLineSegment(p1, p2, width, height, color)
    local dx = p2[1] - p1[1]
    local dy = p2[2] - p1[2]
    local length = math.sqrt(dx*dx + dy*dy)

    if length < 0.001 then return nil end

    -- Create cuboid
    local size = {length, width, height}
    local shape = sim.createPureShape(0, 8, size, 0.001, nil)

    -- Position at midpoint
    local mid_x = (p1[1] + p2[1]) / 2
    local mid_y = (p1[2] + p2[2]) / 2
    local mid_z = height / 2 + 0.001  -- Slightly above ground

    sim.setObjectPosition(shape, -1, {mid_x, mid_y, mid_z})

    -- Rotate to align with the line direction
    local angle = math.atan2(dy, dx)
    sim.setObjectOrientation(shape, -1, {0, 0, angle})

    -- Set color (white for lane markings)
    color = color or {1.0, 1.0, 1.0}
    sim.setShapeColor(shape, nil, sim.colorcomponent_ambient_diffuse, color)

    -- Make static (not affected by physics)
    sim.setObjectInt32Parameter(shape, sim.shapeintparam_static, 1)
    sim.setObjectInt32Parameter(shape, sim.shapeintparam_respondable, 0)

    return shape
end

-- Generate points along a circular arc
function generateArcPoints(center_x, center_y, radius, start_angle, end_angle, num_points)
    local points = {}
    local angle_step = (end_angle - start_angle) / (num_points - 1)

    for i = 0, num_points - 1 do
        local angle = start_angle + i * angle_step
        local x = center_x + radius * math.cos(angle)
        local y = center_y + radius * math.sin(angle)
        table.insert(points, {x, y, 0})
    end

    return points
end

-- Generate points along a straight line
function generateStraightPoints(start_x, start_y, end_x, end_y, num_points)
    local points = {}

    for i = 0, num_points - 1 do
        local t = i / (num_points - 1)
        local x = start_x + t * (end_x - start_x)
        local y = start_y + t * (end_y - start_y)
        table.insert(points, {x, y, 0})
    end

    return points
end

-- Create a continuous line from points
function createSolidLineFromPoints(points, width, height, name)
    local segments = {}

    for i = 1, #points - 1 do
        local segment = createLineSegment(points[i], points[i+1], width, height)
        if segment then
            table.insert(segments, segment)
        end
    end

    -- Group all segments under a dummy
    if #segments > 0 then
        local parent = sim.createDummy(0.05, nil)
        sim.setObjectName(parent, name)

        for _, seg in ipairs(segments) do
            sim.setObjectParent(seg, parent, true)
        end

        return parent
    end

    return nil
end

-- Create dashed line from points
function createDashedLineFromPoints(points, width, height, dash_len, gap_len, name)
    local segments = {}
    local current_dist = 0
    local in_dash = true
    local dash_start_idx = 1

    for i = 1, #points - 1 do
        local p1 = points[i]
        local p2 = points[i+1]
        local dx = p2[1] - p1[1]
        local dy = p2[2] - p1[2]
        local seg_len = math.sqrt(dx*dx + dy*dy)

        if in_dash then
            local segment = createLineSegment(p1, p2, width, height)
            if segment then
                table.insert(segments, segment)
            end

            current_dist = current_dist + seg_len
            if current_dist >= dash_len then
                in_dash = false
                current_dist = 0
            end
        else
            current_dist = current_dist + seg_len
            if current_dist >= gap_len then
                in_dash = true
                current_dist = 0
            end
        end
    end

    -- Group all segments
    if #segments > 0 then
        local parent = sim.createDummy(0.05, nil)
        sim.setObjectName(parent, name)

        for _, seg in ipairs(segments) do
            sim.setObjectParent(seg, parent, true)
        end

        return parent
    end

    return nil
end

-- Create ground plane
function createGround()
    local size = TRACK_PARAMS.ground_size
    local planeSize = {size, size, 0.01}
    local plane = sim.createPureShape(0, 8, planeSize, 0.01, nil)

    sim.setObjectPosition(plane, -1, {0, 0, -0.005})
    sim.setShapeColor(plane, nil, sim.colorcomponent_ambient_diffuse, {0.2, 0.2, 0.2})

    sim.setObjectInt32Parameter(plane, sim.shapeintparam_static, 1)
    sim.setObjectInt32Parameter(plane, sim.shapeintparam_respondable, 0)

    sim.setObjectName(plane, "Ground")

    return plane
end

-- Build complete track for Scenario 1
function buildScenario1Track()
    print("Building Scenario 1 Track (Bing et al. 2018)...")
    print("  Track layout:")
    print("    A: Straight +X (5.0m)")
    print("    B: 90° left turn (r1)")
    print("    C: Straight +Y (5.0m)")
    print("    D: 180° left turn (r1)")
    print("    E: 90° right turn (r2)")
    print("    F: 180° left turn (r1)")

    local width = TRACK_PARAMS.line_width
    local height = TRACK_PARAMS.line_height
    local lane_half_width = TRACK_PARAMS.lane_width / 2  -- 0.25m

    local all_outer_points = {}
    local all_inner_points = {}
    local all_center_points = {}

    -- SECTION A: Straight in +X direction from (0,0) to (5,0)
    print("  Building Section A: Straight +X")
    local a_center = generateStraightPoints(0, 0, 5.0, 0, 20)
    local a_outer = generateStraightPoints(0, lane_half_width, 5.0, lane_half_width, 20)
    local a_inner = generateStraightPoints(0, -lane_half_width, 5.0, -lane_half_width, 20)

    for _, p in ipairs(a_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(a_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(a_inner) do table.insert(all_inner_points, p) end

    -- SECTION B: 90° left turn (r1 radii)
    -- Arc center at (5.0, 2.0), sweep from 270° to 0° (counterclockwise)
    print("  Building Section B: 90° left turn")
    local b_center = generateArcPoints(5.0, 2.0, TRACK_PARAMS.r1_center, -math.pi/2, 0, 30)
    local b_outer = generateArcPoints(5.0, 2.0, TRACK_PARAMS.r1_outer, -math.pi/2, 0, 30)
    local b_inner = generateArcPoints(5.0, 2.0, TRACK_PARAMS.r1_inner, -math.pi/2, 0, 30)

    for _, p in ipairs(b_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(b_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(b_inner) do table.insert(all_inner_points, p) end

    -- SECTION C: Straight in +Y direction from (7,2) to (7,7)
    print("  Building Section C: Straight +Y")
    local c_center = generateStraightPoints(7.0, 2.0, 7.0, 7.0, 20)
    local c_outer = generateStraightPoints(7.0 + lane_half_width, 2.0, 7.0 + lane_half_width, 7.0, 20)
    local c_inner = generateStraightPoints(7.0 - lane_half_width, 2.0, 7.0 - lane_half_width, 7.0, 20)

    for _, p in ipairs(c_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(c_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(c_inner) do table.insert(all_inner_points, p) end

    -- SECTION D: 180° left turn (r1 radii)
    -- Arc center at (5.0, 7.0), sweep from 0° to 180° (counterclockwise)
    print("  Building Section D: 180° left turn")
    local d_center = generateArcPoints(5.0, 7.0, TRACK_PARAMS.r1_center, 0, math.pi, 40)
    local d_outer = generateArcPoints(5.0, 7.0, TRACK_PARAMS.r1_outer, 0, math.pi, 40)
    local d_inner = generateArcPoints(5.0, 7.0, TRACK_PARAMS.r1_inner, 0, math.pi, 40)

    for _, p in ipairs(d_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(d_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(d_inner) do table.insert(all_inner_points, p) end

    -- SECTION E: 90° right turn (r2 radii - WIDER curve)
    -- Arc center at (0, 7.0), sweep from 0° to -90° (clockwise)
    print("  Building Section E: 90° right turn (wider)")
    local e_center = generateArcPoints(0, 7.0, TRACK_PARAMS.r2_center, 0, -math.pi/2, 30)
    -- For right turn: outer boundary has smaller radius, inner has larger radius
    local e_outer = generateArcPoints(0, 7.0, TRACK_PARAMS.r2_outer, 0, -math.pi/2, 30)
    local e_inner = generateArcPoints(0, 7.0, TRACK_PARAMS.r2_inner, 0, -math.pi/2, 30)

    for _, p in ipairs(e_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(e_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(e_inner) do table.insert(all_inner_points, p) end

    -- SECTION F: 180° left turn (r1 radii)
    -- Arc center at (0, 2.0), sweep from 90° to 270° (counterclockwise)
    print("  Building Section F: 180° left turn")
    local f_center = generateArcPoints(0, 2.0, TRACK_PARAMS.r1_center, math.pi/2, 3*math.pi/2, 40)
    local f_outer = generateArcPoints(0, 2.0, TRACK_PARAMS.r1_outer, math.pi/2, 3*math.pi/2, 40)
    local f_inner = generateArcPoints(0, 2.0, TRACK_PARAMS.r1_inner, math.pi/2, 3*math.pi/2, 40)

    for _, p in ipairs(f_center) do table.insert(all_center_points, p) end
    for _, p in ipairs(f_outer) do table.insert(all_outer_points, p) end
    for _, p in ipairs(f_inner) do table.insert(all_inner_points, p) end

    -- Verify track closure
    local start_point = all_center_points[1]
    local end_point = all_center_points[#all_center_points]
    local closure_error = math.sqrt((end_point[1] - start_point[1])^2 + (end_point[2] - start_point[2])^2)
    print(string.format("  Track closure error: %.4f m", closure_error))

    -- Create the track lines
    print("Creating track markings...")
    local outer_line = createSolidLineFromPoints(all_outer_points, width, height, "OuterBoundary")
    local inner_line = createSolidLineFromPoints(all_inner_points, width, height, "InnerBoundary")
    local center_line = createDashedLineFromPoints(all_center_points, width, height,
                                                   TRACK_PARAMS.dash_length,
                                                   TRACK_PARAMS.dash_gap,
                                                   "CenterDashed")

    -- Create ground
    local ground = createGround()

    -- Group everything together
    local track_group = sim.createDummy(0.1, nil)
    sim.setObjectName(track_group, "Track_Scenario1")

    if ground then sim.setObjectParent(ground, track_group, true) end
    if outer_line then sim.setObjectParent(outer_line, track_group, true) end
    if inner_line then sim.setObjectParent(inner_line, track_group, true) end
    if center_line then sim.setObjectParent(center_line, track_group, true) end

    print("Track construction complete!")
    print("  Lane width: " .. TRACK_PARAMS.lane_width .. " m")
    print("  r1 (B,D,F): inner=" .. TRACK_PARAMS.r1_inner .. "m, outer=" .. TRACK_PARAMS.r1_outer .. "m")
    print("  r2 (E): inner=" .. TRACK_PARAMS.r2_inner .. "m, outer=" .. TRACK_PARAMS.r2_outer .. "m")
    print("  Straight sections: " .. TRACK_PARAMS.l1 .. " m")
    print("  Total center points: " .. #all_center_points)

    return track_group
end

-- Main initialization function (called by CoppeliaSim)
function sysCall_init()
    buildScenario1Track()
end

function sysCall_cleanup()
    -- Cleanup code if needed
end
