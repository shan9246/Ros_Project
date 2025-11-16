# autonomous_navigator.py - Fixed version with continuous movement after obstacle avoidance
# Key fixes: Improved obstacle detection reset, better state transitions, continuous navigation

import random
import math
import heapq
from controller import Robot, GPS
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time

class NavigationState(Enum):
    PLANNING = "planning"
    NAVIGATING = "navigating"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    AREA_COMPLETE = "area_complete"
    EMERGENCY_ASCENT = "emergency_ascent"

@dataclass
class GridCell:
    x: int
    y: int
    is_obstacle: bool = False
    is_visited: bool = False
    scan_priority: float = 1.0
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class AdvancedAutonomousNavigator:
    def __init__(self, gps: GPS, world_bounds: Tuple[float, float, float, float] = (0, 20, 0, 20)):
        """
        Advanced autonomous navigator with A* pathfinding and systematic coverage
        
        Args:
            gps: GPS device for position tracking
            world_bounds: (min_x, max_x, min_y, max_y) defining the search area
        """
        self.gps = gps
        self.world_bounds = world_bounds  # (min_x, max_x, min_y, max_y)
        
        # Grid configuration
        self.grid_resolution = 1.0  # 1 meter per grid cell
        self.grid_width = int((world_bounds[1] - world_bounds[0]) / self.grid_resolution) + 1
        self.grid_height = int((world_bounds[3] - world_bounds[2]) / self.grid_resolution) + 1
        
        # Initialize grid
        self.grid = self._initialize_grid()
        self.visited_cells: Set[Tuple[int, int]] = set()
        
        # Navigation state
        self.state = NavigationState.PLANNING
        self.current_path: List[Tuple[int, int]] = []
        self.current_target_index = 0
        self.current_waypoint: Optional[Tuple[float, float]] = None
        
        # Coverage pattern
        self.coverage_pattern = self._generate_systematic_coverage_pattern()
        self.pattern_index = 0
        
        # FIX 1: Improved obstacle avoidance parameters
        self.obstacle_avoidance_height = 2.0
        self.normal_flight_height = 1.0
        self.current_flight_height = self.normal_flight_height
        
        # FIX 2: Better obstacle detection state management
        self.obstacle_detected = False
        self.obstacle_detection_count = 0  # Track consecutive detections
        self.min_detection_count = 2      # Require multiple detections before reacting
        self.obstacle_clear_count = 0     # Track consecutive clear readings
        self.min_clear_count = 3          # Require multiple clear readings before resuming
        
        # FIX 3: Improved timing controls
        self.obstacle_avoidance_timer = 0.0
        self.obstacle_cooldown = 1.5      # Reduced cooldown
        
        # FIX 4: Enhanced horizontal avoidance
        self.horizontal_avoidance_active = False
        self.avoidance_direction = None
        self.avoidance_distance = 2.5     # Reduced distance
        self.avoidance_start_pos = None
        self.avoidance_timeout = 5.0      # Maximum time for horizontal avoidance
        self.avoidance_start_time = 0.0
        
        # Navigation parameters
        self.waypoint_threshold = 0.8
        self.max_speed = 0.4
        self.min_speed = 0.1
        
        # A* pathfinding parameters
        self.diagonal_movement = True
        self.heuristic_weight = 1.2
        
        # FIX 5: Better replanning logic
        self.last_replan_time = 0.0
        self.replan_interval = 0.5        # More frequent replanning
        self.force_replan_flag = False    # Flag to force immediate replanning
        
        # FIX 6: Movement resumption tracking
        self.was_avoiding_obstacle = False
        self.resume_navigation_timer = 0.0
        
        print(f"[NAVIGATOR] Initialized with {self.grid_width}x{self.grid_height} grid")
        print(f"[NAVIGATOR] Coverage pattern: {len(self.coverage_pattern)} waypoints")
    
    def _initialize_grid(self) -> List[List[GridCell]]:
        """Initialize the navigation grid"""
        grid = []
        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                cell = GridCell(x, y)
                cell.scan_priority = 1.0 + random.uniform(-0.2, 0.3)
                row.append(cell)
            grid.append(row)
        return grid
    
    def _world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.world_bounds[0]) / self.grid_resolution)
        grid_y = int((world_y - self.world_bounds[2]) / self.grid_resolution)
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = self.world_bounds[0] + grid_x * self.grid_resolution
        world_y = self.world_bounds[2] + grid_y * self.grid_resolution
        return world_x, world_y
    
    def _generate_systematic_coverage_pattern(self) -> List[Tuple[float, float]]:
        """Generate a systematic coverage pattern for the disaster area"""
        pattern = []
        
        # Boustrophedon (lawnmower) pattern with adaptive spacing
        spacing = 2.0  # 2 meter spacing between passes
        y_positions = []
        
        current_y = self.world_bounds[2] + 1.0
        while current_y < self.world_bounds[3] - 1.0:
            y_positions.append(current_y)
            current_y += spacing
        
        # Create zigzag pattern
        for i, y in enumerate(y_positions):
            if i % 2 == 0:  # Even rows: left to right
                x_start = self.world_bounds[0] + 1.0
                x_end = self.world_bounds[1] - 1.0
                x_positions = [x_start + j * spacing for j in range(int((x_end - x_start) / spacing) + 1)]
            else:  # Odd rows: right to left
                x_start = self.world_bounds[1] - 1.0
                x_end = self.world_bounds[0] + 1.0
                x_positions = [x_start - j * spacing for j in range(int((x_start - x_end) / spacing) + 1)]
            
            for x in x_positions:
                if self.world_bounds[0] <= x <= self.world_bounds[1]:
                    pattern.append((x, y))
        
        # Add perimeter patrol points for thorough coverage
        perimeter_points = self._generate_perimeter_points()
        pattern.extend(perimeter_points)
        
        return pattern
    
    def _generate_perimeter_points(self) -> List[Tuple[float, float]]:
        """Generate points along the perimeter for edge coverage"""
        perimeter = []
        step = 3.0
        
        # Top edge
        x = self.world_bounds[0]
        while x <= self.world_bounds[1]:
            perimeter.append((x, self.world_bounds[3] - 0.5))
            x += step
        
        # Right edge
        y = self.world_bounds[3]
        while y >= self.world_bounds[2]:
            perimeter.append((self.world_bounds[1] - 0.5, y))
            y -= step
        
        # Bottom edge
        x = self.world_bounds[1]
        while x >= self.world_bounds[0]:
            perimeter.append((x, self.world_bounds[2] + 0.5))
            x -= step
        
        # Left edge
        y = self.world_bounds[2]
        while y <= self.world_bounds[3]:
            perimeter.append((self.world_bounds[0] + 0.5, y))
            y += step
        
        return perimeter
    
    def update_obstacle_map(self, current_pos: Tuple[float, float], 
                           range_sensors: dict, detection_threshold: float = 1.0):
        """FIX 7: Improved obstacle detection with hysteresis"""
        current_x, current_y = current_pos
        grid_x, grid_y = self._world_to_grid(current_x, current_y)
        
        # Mark current position as visited
        self.visited_cells.add((grid_x, grid_y))
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid[grid_y][grid_x].is_visited = True
        
        # Check for immediate obstacles
        immediate_obstacle = False
        detected_directions = []
        
        for direction, distance in range_sensors.items():
            if distance < detection_threshold:
                immediate_obstacle = True
                detected_directions.append(direction)
                # Calculate obstacle position
                obstacle_x, obstacle_y = self._calculate_obstacle_position(
                    current_pos, direction, distance
                )
                obs_grid_x, obs_grid_y = self._world_to_grid(obstacle_x, obstacle_y)
                
                # Mark obstacle in grid with buffer zone
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        new_x, new_y = obs_grid_x + dx, obs_grid_y + dy
                        if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                            self.grid[new_y][new_x].is_obstacle = True
        
        # FIX 8: Hysteresis-based obstacle detection
        if immediate_obstacle:
            self.obstacle_detection_count += 1
            self.obstacle_clear_count = 0
            if self.obstacle_detection_count >= self.min_detection_count:
                if not self.obstacle_detected:
                    print(f"[NAVIGATOR] Obstacle confirmed after {self.obstacle_detection_count} detections")
                self.obstacle_detected = True
                self.obstacle_directions = detected_directions
        else:
            self.obstacle_clear_count += 1
            self.obstacle_detection_count = 0
            if self.obstacle_clear_count >= self.min_clear_count:
                if self.obstacle_detected:
                    print(f"[NAVIGATOR] Obstacle cleared after {self.obstacle_clear_count} clear readings")
                    self.was_avoiding_obstacle = True  # Flag that we were avoiding
                self.obstacle_detected = False
        
        return self.obstacle_detected
    
    def _calculate_obstacle_position(self, current_pos: Tuple[float, float], 
                                   direction: str, distance: float) -> Tuple[float, float]:
        """Calculate obstacle position based on sensor direction and distance"""
        x, y = current_pos
        
        if direction == "front":
            return x + distance, y
        elif direction == "back":
            return x - distance, y
        elif direction == "left":
            return x, y + distance
        elif direction == "right":
            return x, y - distance
        else:
            return x, y
    
    def _determine_avoidance_direction(self, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Determine the best direction to avoid obstacles"""
        if not hasattr(self, 'obstacle_directions'):
            return (0.0, 1.0)  # Default: move left
        
        blocked_directions = set(self.obstacle_directions)
        
        # Priority order for avoidance directions - try perpendicular movement first
        avoidance_options = [
            ('right', (0.0, -1.0)),  # Move right
            ('left', (0.0, 1.0)),    # Move left  
            ('back', (-1.0, 0.0)),   # Move backward
            ('front', (1.0, 0.0))    # Move forward (last resort)
        ]
        
        # Choose first unblocked direction
        for direction_name, direction_vector in avoidance_options:
            if direction_name not in blocked_directions:
                return direction_vector
        
        # If all directions blocked, try diagonal movement
        return (0.7, 0.7)
    
    def astar_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm to find optimal path avoiding obstacles"""
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            if self.diagonal_movement:
                dx = abs(a[0] - b[0])
                dy = abs(a[1] - b[1])
                return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
            else:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(node: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = node
            neighbors = []
            
            if self.diagonal_movement:
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            else:
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                    not self.grid[ny][nx].is_obstacle):
                    neighbors.append((nx, ny))
            
            return neighbors
        
        # A* algorithm implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2:
                    tentative_g_score = g_score[current] + math.sqrt(2)
                else:
                    tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic_weight * heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def get_next_target_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get the next target waypoint from the coverage pattern"""
        if self.pattern_index >= len(self.coverage_pattern):
            return None
        
        target = self.coverage_pattern[self.pattern_index]
        self.pattern_index += 1
        return target
    
    def compute_navigation_control(self, current_pos: Tuple[float, float], 
                                 current_time: float) -> Tuple[float, float, float]:
        """
        FIX 9: Main navigation control with improved state transitions
        
        Returns:
            (forward_velocity, sideways_velocity, height_adjustment)
        """
        current_x, current_y = current_pos
        
        # FIX 10: Handle horizontal avoidance with timeout
        if self.horizontal_avoidance_active:
            if self.avoidance_start_pos is None:
                self.avoidance_start_pos = current_pos
                self.avoidance_start_time = current_time
            
            # Check timeout or completion conditions
            avoidance_time = current_time - self.avoidance_start_time
            avoidance_distance_moved = self._distance_to_waypoint(current_pos, self.avoidance_start_pos)
            
            if (avoidance_distance_moved >= self.avoidance_distance or 
                avoidance_time >= self.avoidance_timeout or
                not self.obstacle_detected):  # Exit if obstacle cleared
                
                print(f"[NAVIGATOR] Horizontal avoidance complete (moved: {avoidance_distance_moved:.1f}m, time: {avoidance_time:.1f}s)")
                self.horizontal_avoidance_active = False
                self.avoidance_start_pos = None
                self.force_replan_flag = True  # Force immediate replanning
                self.current_flight_height = self.normal_flight_height
            else:
                # Continue horizontal avoidance
                if self.avoidance_direction:
                    speed = 0.3
                    return (self.avoidance_direction[0] * speed, 
                           self.avoidance_direction[1] * speed, 
                           0.0)
        
        # FIX 11: Handle vertical obstacle avoidance with better exit conditions
        if self.state == NavigationState.AVOIDING_OBSTACLE:
            avoidance_duration = current_time - self.obstacle_avoidance_timer
            
            if (avoidance_duration > self.obstacle_cooldown or not self.obstacle_detected):
                print(f"[NAVIGATOR] Vertical obstacle avoidance complete (duration: {avoidance_duration:.1f}s)")
                self.state = NavigationState.PLANNING
                self.current_flight_height = self.normal_flight_height
                self.force_replan_flag = True
            else:
                # Continue ascending or maintain elevated height
                height_adjustment = max(0, self.obstacle_avoidance_height - self.current_flight_height)
                return 0.0, 0.0, height_adjustment
        
        # FIX 12: Improved obstacle detection handling
        if (self.obstacle_detected and 
            not self.horizontal_avoidance_active and 
            self.state != NavigationState.AVOIDING_OBSTACLE):
            
            print("[NAVIGATOR] Active obstacle detected! Choosing avoidance strategy...")
            
            # Prefer horizontal avoidance for front obstacles
            if hasattr(self, 'obstacle_directions') and 'front' in self.obstacle_directions:
                print("[NAVIGATOR] Front obstacle - starting horizontal avoidance")
                self.horizontal_avoidance_active = True
                self.avoidance_direction = self._determine_avoidance_direction(current_pos)
                self.avoidance_start_pos = current_pos
                self.avoidance_start_time = current_time
                
                # Start avoidance immediately
                speed = 0.3
                return (self.avoidance_direction[0] * speed, 
                       self.avoidance_direction[1] * speed, 
                       0.0)
            else:
                # Use vertical avoidance for complex situations
                print("[NAVIGATOR] Complex obstacle - starting vertical avoidance")
                self.state = NavigationState.AVOIDING_OBSTACLE
                self.obstacle_avoidance_timer = current_time
                self.current_flight_height = self.obstacle_avoidance_height
                return 0.0, 0.0, self.obstacle_avoidance_height - self.normal_flight_height
        
        # FIX 13: Handle post-avoidance navigation resumption
        if self.was_avoiding_obstacle and not self.obstacle_detected:
            print("[NAVIGATOR] Resuming normal navigation after obstacle clearance")
            self.was_avoiding_obstacle = False
            self.force_replan_flag = True
            self.resume_navigation_timer = current_time
        
        # FIX 14: Force replanning when needed
        if (self.force_replan_flag or 
            (self.obstacle_detected and current_time - self.last_replan_time > self.replan_interval)):
            
            print("[NAVIGATOR] Forcing replan due to changed conditions")
            self.force_replan()
            self.last_replan_time = current_time
            self.force_replan_flag = False
        
        # FIX 15: Improved waypoint management
        if (self.current_waypoint is None or 
            self._distance_to_waypoint(current_pos, self.current_waypoint) < self.waypoint_threshold):
            
            # Get next waypoint
            next_waypoint = self.get_next_target_waypoint()
            if next_waypoint is None:
                print("[NAVIGATOR] Area coverage complete!")
                self.state = NavigationState.AREA_COMPLETE
                return 0.0, 0.0, 0.0
            
            self.current_waypoint = next_waypoint
            self.state = NavigationState.PLANNING
            print(f"[NAVIGATOR] New target: ({next_waypoint[0]:.1f}, {next_waypoint[1]:.1f})")
        
        # Plan path using A* if needed
        if self.state == NavigationState.PLANNING:
            start_grid = self._world_to_grid(current_x, current_y)
            goal_grid = self._world_to_grid(self.current_waypoint[0], self.current_waypoint[1])
            
            self.current_path = self.astar_pathfind(start_grid, goal_grid)
            
            if not self.current_path:
                print("[NAVIGATOR] No path found, trying next waypoint")
                self.current_waypoint = None
                return 0.0, 0.0, 0.0
            
            self.current_target_index = 0
            self.state = NavigationState.NAVIGATING
            print(f"[NAVIGATOR] New path planned with {len(self.current_path)} waypoints")
        
        # Navigate along planned path
        if self.state == NavigationState.NAVIGATING and self.current_path:
            # Get current target from path
            if self.current_target_index < len(self.current_path):
                target_grid = self.current_path[self.current_target_index]
                target_world = self._grid_to_world(target_grid[0], target_grid[1])
                
                # Check if we've reached current path target
                if self._distance_to_waypoint(current_pos, target_world) < self.waypoint_threshold:
                    self.current_target_index += 1
                
                # Calculate movement towards target
                if self.current_target_index < len(self.current_path):
                    target_grid = self.current_path[self.current_target_index]
                    target_world = self._grid_to_world(target_grid[0], target_grid[1])
                    return self._calculate_movement_control(current_pos, target_world)
        
        # Default: move towards final waypoint
        if self.current_waypoint:
            return self._calculate_movement_control(current_pos, self.current_waypoint)
        
        return 0.0, 0.0, 0.0
    
    def _distance_to_waypoint(self, current_pos: Tuple[float, float], 
                             waypoint: Tuple[float, float]) -> float:
        """Calculate distance to waypoint"""
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        return math.sqrt(dx**2 + dy**2)
    
    def _calculate_movement_control(self, current_pos: Tuple[float, float], 
                                  target_pos: Tuple[float, float]) -> Tuple[float, float, float]:
        """Calculate movement control commands to reach target position"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return 0.0, 0.0, 0.0
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Adaptive speed based on distance and conditions
        base_speed = min(self.max_speed, max(self.min_speed, distance * 0.5))
        
        # Reduce speed slightly if we were recently avoiding obstacles
        if self.was_avoiding_obstacle and time.time() - self.resume_navigation_timer < 2.0:
            base_speed *= 0.8
        
        forward_velocity = base_speed * dx
        sideways_velocity = base_speed * dy
        
        # Height adjustment
        height_adjustment = 0.0
        if self.current_flight_height != self.normal_flight_height:
            height_adjustment = (self.current_flight_height - self.normal_flight_height) * 0.1
        
        return forward_velocity, sideways_velocity, height_adjustment
    
    def get_coverage_progress(self) -> dict:
        """Get current coverage progress statistics"""
        total_cells = self.grid_width * self.grid_height
        visited_cells = len(self.visited_cells)
        
        progress = {
            'total_waypoints': len(self.coverage_pattern),
            'completed_waypoints': self.pattern_index,
            'progress_percentage': (self.pattern_index / len(self.coverage_pattern)) * 100,
            'grid_coverage': (visited_cells / total_cells) * 100,
            'current_state': self.state.value,
            'current_target': self.current_waypoint,
            'obstacle_detected': self.obstacle_detected,
            'avoidance_active': self.horizontal_avoidance_active or (self.state == NavigationState.AVOIDING_OBSTACLE)
        }
        
        return progress
    
    def force_replan(self):
        """Force replanning of current path"""
        self.state = NavigationState.PLANNING
        self.current_path = []
        self.current_target_index = 0
        print("[NAVIGATOR] Forced replanning initiated")
    
    def is_mission_complete(self) -> bool:
        """Check if the area coverage mission is complete"""
        return self.state == NavigationState.AREA_COMPLETE
    
    def get_current_flight_height(self) -> float:
        """Get the current target flight height"""
        return self.current_flight_height